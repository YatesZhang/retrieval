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
from transformers import CLIPModel
# from copy import deepcopy


class Runner(object):
    def __init__(self, 
                 model: torch.nn.Module, 
                 train_loader: Optional[DataLoader], 
                 test_loader: Optional[DataLoader], 
                 optimizer: Optimizer, 
                 scheduler,
                 logger: Recorder, 
                 workflows: list) -> None:
        # dataloader
        self.test_loader = test_loader
        self.train_loader = train_loader
        if self.test_loader is None:
            self.test_loader = set_drop_last(self.train_loader,drop_last=False)

        # Auto-deployment on GPU:
        accelerator = Accelerator()
        """
            Attention:
                accelerator will change the collate_fn in dataloader
        """
        model, optimizer, self.train_loader, self.test_loader, scheduler = \
            accelerator.prepare(model, optimizer, self.train_loader, self.test_loader,scheduler)

        # prompt tuning
        if not isinstance(model, CLIPModel):
            model.prompt_tuning_init(logger)
        self.model = model 
        # on GPU
        self.model = DataParallel(self.model).cuda()
        # self.model = self.model.cuda()



        # optimizer
        self.optimizer = optimizer 
        self.scheduler = scheduler 

        # utils
        self.logger = logger 
        self.accelerator = accelerator

        # common: 
        self.workflows = workflows
        self.train_epoch = 0
        self.totol_epochs = self.get_totol_epochs()
        
        # path
        self.work_dir = self.logger.work_dir
        self.checkpoint_dir = os.path.join(self.work_dir, 'checkpoint')

        # buffer
        self.test_gt = None 
        self.queries_gt = None

        # metric
        
    def get_model(self):
        if isinstance(self.model, DataParallel):
            return self.model.module
        elif isinstance(self.model, torch.nn.Module):
            return self.model
        
    def get_totol_epochs(self):
        totol_epochs = 0
        for flow, epochs in self.workflows:
            if flow == 'train':
                totol_epochs += epochs 
        return totol_epochs 
    
    def run(self):
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

    def evaluate(self, I: np.ndarray, cls_topK: dict):
        en_cls_cnt = self.test_loader.dataset.en_cls_cnt
        
        # self.logger.info(cls_topK, title="CLASS TOP K")
        # self.logger.info(en_cls_cnt, title="number of samples per class")

        result_TP = {k:0 for k in cls_topK}
        result_GT = {k:0 for k in cls_topK}
        result_Positive = {k:0 for k in cls_topK}
        result_Recall = {k:[] for k in cls_topK}
        result_Precision = {k:[] for k in cls_topK}
        # pdb.set_trace()
        for i, q_gt in enumerate(self.queries_gt):
            # top K samples are considered to be positive (top K == FP + TP)
            top_k = cls_topK[q_gt]   
            rank = I[i]
            rank = rank[:top_k]

            # TP:
            _TP = (self.test_gt[rank] == q_gt).astype('int').sum() 

            # GT:
            num_gt = en_cls_cnt[q_gt]    # ground truth in test set
            
            # update results:
            result_TP[q_gt] += _TP
            result_GT[q_gt] += num_gt
            result_Positive[q_gt] += top_k

            # Recall & Precision
            recall = _TP / num_gt        #
            precision = _TP / top_k      # 
            result_Recall[q_gt].append(recall)
            result_Precision[q_gt].append(precision)
        
        
        mAP = 0
        for _cls in cls_topK:
            top_k = cls_topK[_cls]
            Positive = result_Positive[_cls]
            Recall = result_Recall[_cls]
            Precision = result_Precision[_cls] 
            _TP = result_TP[_cls]    # TP: UnboundLocalError
            GT = result_GT[_cls]

            # accumulate mAP
            AP = voc_ap(Precision, Recall)
            mAP += AP
            
            # update log:
            log_str = "Evaluate on category {:<10},".format(_cls)
            log_str += "AP: {:.3f}%, ".format(AP * 100)
            log_str += "metric: Recall@{}={:3.3f}%  Precision@{:3.3f}%, ".format(top_k, _TP / GT * 100, _TP / Positive * 100)
            log_str += f"with GT: {GT}, TP: {_TP}"
            self.logger.info(log_str)
        mAP /= len(cls_topK)
        self.logger.info("mAP: {:.3f}%".format(mAP * 100))


    def test(self):
        self.before_test_epoch()

        # run single GPU test:
        # -------Generate Embeddings in test set---------
        assert isinstance(self.test_loader.dataset, CLSDataset)
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

        # ----------Generate Queries Embeddings---------
        queries = []
        queries_gt = []
        self.test_loader.dataset.query(True)
        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                images = batch['pixel_values']
                # get embedding & frobenius norm to 1
                out = self.model.module.get_image_features(images)
                out /= out.norm(p=2,dim=-1, keepdim=True)
                queries.append(out)
                if self.queries_gt is None or len(self.queries_gt)==0:
                    queries_gt += [dataset.map_zh2en(text) for text in batch['text']]
                self.after_query_step(step=step)    
        queries = torch.cat(queries, dim=0)    # [N_q, 512]
        queries_gt = np.array(queries_gt)

        # -----------------SEARCH------------------------
        gpu_index = faiss.index_cpu_to_all_gpus(   # build the index
            faiss.IndexFlatL2(queries.shape[1])    # d: dimension == 512
        )

        # add embeddings:
        gpu_index.add(embeds.cpu().numpy())

        # get top K for every class:
        cls_topK = self.test_loader.dataset.en_cls_cnt
        """
        e.g.
            cls_topK=dict(
                tricycle=1000,
                truck=2000,
                bus=1000,
                motorbike=1000            
            )
            Kmax=2000
        """
        Kmax = max([cls_topK[k] for k in cls_topK])
        num_samples_test = sum([cls_topK[k] for k in cls_topK])
        """
            I: np.ndarray [N_{image_queries}, Kmax]
            device: cpu
            _: L2 Distance is ignored
        """
        I = None
        try:
            # C++ Error encountered when Kmax is a Variable, not a Const
            # use int(Kmax) to cast the Kmax to c++ const value
            _, I = gpu_index.search(queries.cpu().numpy(), int(Kmax))
        except TypeError:    # not a const value
            import pdb 
            pdb.set_trace()
        # -----------------EVALUATE---------------------
             
        if self.test_gt is None or len(self.test_gt) == 0:
            self.test_gt = test_gt
            assert len(self.test_gt) == len(embeds)
        if self.queries_gt is None or len(self.queries_gt) == 0:
            self.queries_gt = queries_gt
            assert len(self.queries_gt) == len(I) and len(I) == len(queries)
            # self.queirs_gt = np.array(self.queirs_gt)
        
        self.evaluate(I, cls_topK)
        for K in [100, 50, 20, 10, 5, 1]:
            cls_topK = {_cls: K for _cls in cls_topK}
            self.evaluate(I, cls_topK)

        



