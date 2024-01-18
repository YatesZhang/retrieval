import os 
import os.path as osp
from tqdm import tqdm
from rich import print
import torch
from torch.utils.data import DataLoader
import torch.mutiprocessing as mp
from Flamingo.structure.Detection import Detection2CLSLabel


class CacheCLIPOutRunner(object):
    """ 
        deepspeed inferencer for clip model
    """
    def __init__(self,
                clip_model,
                batch_processor,
                cache_dir,
                dataset,
                batch_size_per_gpu=4,
                mp_size=4,   # model parallel size
                collate_fn=None):
        self.batch_processor = batch_processor
        self.batch_size_per_gpu = batch_size_per_gpu

        # cache dir:
        self.cache_dir = cache_dir
        # pth file dir and annotation file will be created in init_cache_dir()
        self.pth_dir = None    
        """ 
            annotation file: .csv

            cache_dir:
            |-- annotations
            |   |-- annotations.csv
            |-- pth
            |   |-- *.pth
        """
        self.annotation_file = None    
        # rank zero only:
        self.init_cache_dir()

        # cache data infos:
        """ 
            data_infos: [
                dict(
                    img_path=...,
                    bbox=...,
                )
            ]
        """

        # set mp_size:
        print("muti-threading degree: mp_size={}".format(mp_size))
        self.mp_size = mp_size

        # init dataloader for each rank:
        self.dataloaders = []
        self.init_dataloader(dataset, collate_fn=collate_fn)
        
        # init model for each rank:
        self.clip_models = []
        self.init_models(clip_model)

        # output annotaion 
        self.out_results = [
            [] for _ in range(mp_size)
        ]
    
    def init_dataloader(self, dataset, collate_fn=None):
        # init dataset for each rank:
        assert type(dataset).__name__ in ['ParticipantsProperty']
        datasets = dataset.split(self.mp_size)
        self.dataloaders = []
        for rank in range(self.mp_size):
            dataloader = DataLoader(dataset=datasets[rank],
                                    batch_size=self.batch_size_per_gpu,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=collate_fn)
            self.dataloaders.append(dataloader)
            print("init dataloader for rank {}".format(rank))
    
    def init_models(self, clip_model):
        """ 
            init model for each GPU
        """
        print("init clip model for each rank, set to cuda with mode of evaluation...")
        self.clip_models = [
            clip_model.cuda(rank).eval() for rank in range(self.mp_size)
        ]

    @torch.no_grad()
    def thread_worker(self, rank):
        """ 
            thread worker for muti gpu inference
        """
        # load model:
        clip_model = self.clip_models[rank]
        dataloader = self.dataloaders[rank]
        for data in tqdm(dataloader):
            assert isinstance(data, Detection2CLSLabel)
            imgs = data["img"]
            out = self.batch_processor(clip_model, imgs)

    def init_cache_dir(self):
        """ 
            make cache dir if not exists
        """
        if not osp.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
            print("make cache dir:", self.cache_dir)
        else:
            print("cache dir already exists:", self.cache_dir)
        
        # cache .pth file:
        self.pth_dir = osp.join(self.cache_dir, "pth")
        if not osp.exists(self.pth_dir):
            os.mkdir(self.pth_dir)
            print("make pth dir:", self.pth_dir)
        else:
            print("pth dir already exists:", self.pth_dir) 
        
        # cache annotation file:
        annotation_dir = osp.join(self.cache_dir, "annotations")
        if not osp.exists(annotation_dir):
            os.mkdir(annotation_dir)
            print("make annotation dir:", annotation_dir)
        else:
            print("annotation dir already exists:", annotation_dir)
        
        self.annotation_file = osp.join(annotation_dir, "annotations.csv")
        print("annotation will be cached at: ", self.annotation_file)

    def run(self):
        for data in tqdm(self.dataloader):
            for rank in range(self.mp_size):
                p = mp.Process(target=self.thread_worker, args=(rank, data))
        