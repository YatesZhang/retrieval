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
        totol_epochs = 0
        for flow, epochs in self.workflows:
            if flow == 'train':
                totol_epochs += epochs 
        return totol_epochs
    
    @rank_zero_only
    def before_run(self):
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
    def info(self, msg):
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
            self.info(f"WORKFLOW: {flow}, EPOCHS: {epochs}")
            for _ in range(epochs):
                if flow == 'train':
                    self.train_epoch += 1
                workflow_fn()
        return 
    
    def resume(self, path: Optional[str] = None, load_optim=False):
        """ 
            resume from checkpoint: 
        """
        if path is None or path == '':
            return 
        if not os.path.exists(path):
            raise FileNotFoundError
        checkpoint = torch.load(path)
        self.get_model().load_state_dict(checkpoint['model'])
        self.logger.info(f"load model from: {path}[model]")
        if load_optim:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info(f"load optimizer from {path}[optimizer]")
        else:
            self.logger.info("optimizer is not loaded")
        return
     
    def save_checkpoint(self):
        return 
    
    def before_test_epoch(self):
        """ 
            before test epoch hook
        """
        self.model.eval()
        if not self.test_loader.dataset.test_mode:
            self.test_loader.dataset.train(False)
        if self.test_loader.drop_last:
            self.test_loader.drop_last = False
        self.logger.info("Start Running Test:")
        
    def before_train_epoch(self):
        self.step = 0
        self.model.train()

    
    def after_train_epoch(self):
        if self.train_epoch == 1 or self.train_epoch == self.total_epochs or self.train_epoch % 4 == 0:
            self.save_checkpoint()
        return 
    
    def after_test_step(self, step):
        """ 
            after test step hook
        """
        if step % 10 == 0:
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
            self.info("[rank@{rank}|{world_size}][Epoch:{epoch}|{total_epoch}][Step:{step}|{total_step}] \
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
        self.before_test_epoch()

        # run single GPU test:
        # -------Generate Embeddings in test set---------
        dataset = self.test_loader.dataset   
        embeds = []
        test_gt = []
        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                images = batch['pixel_values']
                # get embedding & frobenius norm to 1
                out = self.model.module.get_image_features(images)
                out /= out.norm(p=2,dim=-1, keepdim=True)
                embeds.append(out)
                if self.test_gt is None or len(self.test_gt) == 0:
                    test_gt += [dataset.map_zh2en(text) for text in batch['text']]
                self.after_test_step(step=step)
        embeds = torch.cat(embeds, dim=0)    # [N_t, 512]
        test_gt = np.array(test_gt)

        



