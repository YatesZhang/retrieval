import time
import cv2
import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
import random
import os
from typing import Optional
import pdb 
import deepspeed
from Flamingo.utils.pretty import vis_model
from loguru import logger
import os.path as osp
import datetime
from Flamingo.utils.distributed import rank_zero_only
from Flamingo.utils.utils import get_lora_weight_only

""" 
    1) training step on logger 
    2) save checkpoints on rank 0
    3) evaluation on rank 0 with deepspeed inference
    see: https://github.com/microsoft/DeepSpeed/issues/4287
    4) resume from checkpoint
"""
class Runner(object):
    def __init__(self,
                model: torch.nn.Module,
                train_dataloader: DataLoader,
                test_dataloader: DataLoader,
                batch_processor=None, 
                optimizer=None,
                lr_scheduler=None, 
                workflows=[('train', 1), ('test', 1)],
                args=None) -> None:
        """ 
            Runner for DeeoSpeed training
        """
        # DeepSpeed config:
        self.args = args
        self.zero_stage = args.zero_stage
        
        # Distributed config:
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        
        # get work flows:
        self.workflows = workflows
        self.train_epoch = 0
        self.total_epochs = self.get_totol_epochs()
        self.total_steps = args.num_update_steps_per_epoch    # a step means a backward call
        self.train_dataset_name = type(self.train_dataloader.dataset).__name__

        # batch processor: do forward pass and return loss 
        self.batch_processor = batch_processor

        # init logger:
        self.work_dir = args.work_dir
        self.init_logger()

        # init engine
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=args.ds_config 
        )

        self.model = model 
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # init buffer: 
        self.loss = None 
        self.step = -1
    
    @rank_zero_only
    def init_logger(self):                  
        """ 
            init logger at rank 0 only
        """
        # get time:
        _time = datetime.datetime.now().strftime("[%Y-%m-%d][%H:%M:%S]")
        # get pid:
        _time += "PID:{}".format(os.getpid())
        # get log file name:
        logger_name = osp.join(self.work_dir, _time + ".log")
        
        # create work_dir and logger:
        self.logger = logger 
        self.logger.add(logger_name)

        # info:
        self.logger.info("create work dir: {}".format(self.work_dir))
        self.logger.info("create logger: {}".format(logger_name))
        self.logger.info("init logger success!")

    def get_totol_epochs(self):
        """ 
            total epochs
        """
        totol_epochs = 0
        for flow, epochs in self.workflows:
            if flow == 'train':
                totol_epochs += epochs 
        return totol_epochs
    
    @rank_zero_only
    def before_run(self):
        """ 
            before run hook
        """
        # model visualization is integrated in create_model_and_transformers
        # _ = vis_model(self.model)

        # record to disk: 
        self.logger.info(self.model)
        self.logger.info(str(self.optimizer))
        self.logger.info(str(self.lr_scheduler))
        self.logger.info(str(self.workflows))
        # self.logger.info()
        return 
    
    @rank_zero_only
    def info_rank_zero(self, msg):
        """ 
            print info on rank 0 only 
        """
        self.logger.info(msg)

    def run(self):
        """ 
            run workflows
        """
        self.before_run()
        for flow, epochs in self.workflows:
            assert flow in ['train', 'test']
            workflow_fn = getattr(self, flow)
            self.info_rank_zero(f"WORKFLOW: {flow}, EPOCHS: {epochs}")
            for _ in range(epochs):
                if flow == 'train':
                    self.train_epoch += 1
                workflow_fn()
        return 
    
    def resume(self, path: Optional[str] = None, load_optim=False):
        """ 
            resume from checkpoint:
            activated in each rank 
        """
        if path is None or path == '':
            # self.work_dir
            return 
        if not os.path.exists(path):
            raise FileNotFoundError
        
        # load state_dict:
        loRA_weight = torch.load(path)
        self.model.load_state_dict(loRA_weight, strict=False)
        self.logger.info("[rank@{rank}|{world_size}] load from checkpoint {path}",
                         rank=self.rank,
                         world_size=self.world_size,
                         path=path)
        if load_optim:
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.logger.info(f"load optimizer from {path}[optimizer]")
            raise NotImplementedError
        else:
            self.logger.info("[rank@{rank}|{world_size}] optimizer is not loaded !",
                             rank=self.rank,
                             world_size=self.world_size)
        return
    
    @rank_zero_only
    def save_checkpoint(self):
        """ 
            only save loRA weight in rank 0
        """
        loRA_dict = get_lora_weight_only(self.model)
        save_dir = osp.join(self.work_dir, str(self.train_epoch))
        if not osp.exists(save_dir):
            os.mkdir(save_dir)
        weight_path = osp.join(save_dir, 'loRA.pth')
        torch.save(loRA_dict, weight_path)
        self.info_rank_zero("[rank@{rank}|{world_size}]LoRA weight saved at: {weight_path}",
                   weight_path=weight_path,
                   rank=self.rank,
                   world_size=self.world_size)
        return 
    
    def before_test_epoch(self):
        """ 
            before test epoch hook
        """
        self.model.eval()
        self.logger.info("[rank@{rank}|{world_size}] Start Running Test:",
                         rank=self.rank,
                         world_size=self.world_size)
        
    def before_train_epoch(self):
        """ 
            before train hook
        """
        self.step = 0
        self.model.train()
        self.logger.info("[rank@{rank}|{world_size}][@runner.before_train_epoch] set step=0, set model.train()",
                         rank=self.rank,
                         world_size=self.world_size)

    
    def after_train_epoch(self):
        """ 
            after train hook
            1) save loRA weight
        """
        if self.train_epoch == 1 or self.train_epoch == self.total_epochs or self.train_epoch % 1 == 0:
            self.save_checkpoint()
        return 
    
    def after_test_step(self):
        """ 
            after test step hook
        """
        if self.step % 10 == 0:
            self.logger.info("[Step:{:<3}|{}] Generate Embeddings for Image in Test Set".format(str(step),len(self.test_loader)))
        pass

    def after_test_epoch(self):
        """ 
            after test epoch hook
        """
        pass
    
    def before_train_iter(self):
        """ 
            before train iter hook
        """
        pass 
    
    # @rank_zero_only
    def after_train_iter(self):
        """ 
            after train iter hook
        """
        if self.step % 10 == 0:
            self.info_rank_zero("[rank@{rank}|{world_size}][Epoch:{epoch}|{total_epoch}][Step:{step}|{total_step}] \
                      Dataset: {dataset}, Loss: {loss}", 
                epoch=self.train_epoch,
                total_epoch=self.total_epochs,
                rank=self.rank, 
                world_size=self.world_size,
                step=self.step,
                total_step=self.totol_steps,
                loss=self.loss.item(),
                dataset=self.train_dataset_name)
        return  
    
    def call_backward(self):
        """ 
            gradient accumulation has been integraed in DeepSpeed
            model, optimizer are wrapped in DeepSpeed

            if you want to change backward pass to normal torch style:
            use: 
                self.loss.backward()
                self.scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()
        """
        self.model.backward(self.loss)
        self.model.step() 
        return  
    
    # @logger.catch
    def train(self):
        """ 
            train workflow
        """
        self.before_train_epoch()
        for step, batch in enumerate(self.train_dataloader):
            self.step = step
            self.before_train_iter()
            self.loss = self.batch_processor(model=self.model,
             batch=batch, mode='train') 
            self.call_backward() 
            self.after_train_iter() 
        self.after_train_epoch()
        return 

    def test(self):
        """ 
            test phase 
            how to do test in DeepSpeed ? 
        """
        self.before_test_epoch()
        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                self.step = step 
                output = self.batch_processor(model=self.model, batch=batch, mode='test') 
                self.after_test_step()
        raise NotImplementedError

        



