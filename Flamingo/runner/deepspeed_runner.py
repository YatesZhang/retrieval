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
# from copy import deepcopy


class Runner(object):
    def __init__(self, model: torch.nn.Module, datasets: list, workflows: list) -> None:
        deepspeed.init_distributed()
        self.model = model 
        self.workflows = workflows

    def get_totol_epochs(self):
        totol_epochs = 0
        for flow, epochs in self.workflows:
            if flow == 'train':
                totol_epochs += epochs 
        return totol_epochs
     
    def before_run(self):
        pass 
    
    def run(self):
        self.before_run()
        for flow, epochs in self.workflows:
            assert flow in ['train', 'test']
            workflow_fn = getattr(self, flow)
            self.logger.info(f"WORKFLOW: {flow}, EPOCHS: {epochs}")
            for _ in range(epochs):
                if flow == 'train':
                    self.train_epoch += 1
                workflow_fn()
        return 
    
    def resume(self, path: Optional[str] = None, load_optim=False):
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
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        save_path = os.path.join(self.checkpoint_dir, f"clip_{self.train_epoch}.pt")
        torch.save({
            'model': self.get_model().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            }, save_path)
        self.logger.info(f"save model, optimizer at: {save_path}")
        return 
    
    def before_test_epoch(self):
        self.model.eval()
        if not self.test_loader.dataset.test_mode:
            self.test_loader.dataset.train(False)
        if self.test_loader.drop_last:
            self.test_loader.drop_last = False
        self.logger.info("Start Running Test:")
        
    def before_train_epoch(self):
        self.logger.totol_epochs = self.totol_epochs
        self.logger.train_epoch = self.train_epoch
        self.logger.totole_steps = len(self.train_loader) 
        self.model.train()
        if self.train_loader.dataset.test_mode:
            self.train_loader.dataset.train(True)
        if isinstance(self.model, DataParallel):
            self.train_loader.drop = True
    
    def after_train_epoch(self):
        if self.train_epoch == 1 or self.train_epoch == self.totol_epochs or self.train_epoch % 4 == 0:
            self.save_checkpoint()
        pass 
    
    def after_train_step(self, step, loss):
        if step % 10 == 0:
            self.logger.write_loss(step=step, loss=loss, lr=self.scheduler.get_lr()[0])
        pass 

    def after_test_step(self, step):
        if step % 10 == 0:
            self.logger.info("[Step:{:<3}|{}] Generate Embeddings for Image in Test Set".format(str(step),len(self.test_loader)))
        pass
    
    def after_query_step(self, step):
        if step % 10 == 0:
            self.logger.info("[Step:{:<3}|{}] Generate Embeddings for Image Queris".format(str(step),len(self.test_loader)))

    def after_test_epoch(self):
        pass

    def train(self):
        self.before_train_epoch()
        for step, batch in enumerate(self.train_loader):
            # to_cuda(batch)
            outputs = self.model(**batch, return_loss=True)
            # pdb.set_trace()
            loss = outputs.loss.sum()
            # loss.backward()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.after_train_step(step=step, loss=loss)
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

        



