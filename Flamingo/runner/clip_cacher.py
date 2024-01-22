import os 
import json
import os.path as osp
from tqdm import tqdm
from rich import print
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import torch.multiprocessing as mp
from Flamingo.structure.Detection import Detection2CLSLabel, merge_detected
from Flamingo.models.batchprocessor import CLIPBatchProcessor
def collate_fn(batch):
    """ 
        collater function for clip dataloader
        merge data info:
    """
    batch = [Detection2CLSLabel(data_info) for data_info in batch]
    return merge_detected(batch)

class CacheCLIPOutRunner(object):
    """ 
        deepspeed inferencer for clip model
    """
    def __init__(self,
                batch_processor,
                cache_dir,
                dataset,
                batch_size_per_gpu=4,
                mp_size=1,   # model parallel size
                split='train'):
        self.cats = dataset.cats
        print("categories: ", self.cats)
        assert split in ['train', 'val', 'test']
        self.split = split
        print("cache {} set".format(self.split))
        
        assert isinstance(batch_processor, CLIPBatchProcessor)
        self.batch_size_per_gpu = batch_size_per_gpu

        # cache dir:
        self.cache_dir = cache_dir
        # pth file dir and annotation file will be created in init_cache_dir()
        self.pth_dir = None    
        """ 
            annotation file: .json

            cache_dir:
            |-- annotations
            |   |-- train.json
            |   |-- val.json
            |-- pth
            |   |-- category_name
            |       |-- *.pth
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
        num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES").split(','))
        if num_gpus > 1:
            mp_size = num_gpus
        print("muti-threading degree: mp_size={}".format(mp_size))
        self.mp_size = mp_size

        # init dataloader for each rank:
        self.dataloaders = []
        self.init_dataloader(dataset, collate_fn=collate_fn)
        
        # init model for each rank:
        self.clip_models = []
        self.init_models(batch_processor)

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
                                    # num_workers=4,
                                    collate_fn=collate_fn)
            self.dataloaders.append(dataloader)
            print("init dataloader for rank {}".format(rank))
    
    def init_models(self, batch_processor):
        """ 
            init model for each GPU
        """
        print("init clip model for each rank, set to cuda with mode of evaluation...")
        self.clip_models = [
            deepcopy(batch_processor).to(torch.device("cuda:{}".format(rank))) for rank in range(self.mp_size)
        ]
        for rank in range(self.mp_size):
            print("clip model devices:{}".format(str(self.clip_models[rank].device)))
    @torch.no_grad()
    def thread_worker(self, rank):
        """ 
            thread worker for muti gpu inference
        """
        # load model:
        print("thread_worker@rank{} | start inference...".format(rank))
        clip_model = self.clip_models[rank]
        dataloader = self.dataloaders[rank]
        for data in tqdm(dataloader):
            assert isinstance(data, dict)
            imgs = data['imgs']    # List[PIL.Image.Image]
            # import pdb; pdb.set_trace()
            out = clip_model(imgs)    # [B, 1, 1, V, D]
            
            # save to disk
            metas = data['metas']
            category_names = data['category_names']
            attributes_names = data['attributes_names']
            for i in range(len(metas)):
                meta = metas[i]
                instance_id = meta['instance_id']
                area = meta['area']
                bbox = meta['bbox']
                file_name = meta['file_name']
                img_path = meta['img_path']

                category_name = category_names[i]
                attributes_name = attributes_names[i]
                
                # get file name:
                save_dir = osp.join(self.pth_dir, category_name.replace(" ", "_"))
                if not osp.exists(save_dir):
                    os.mkdir(save_dir)
                save_name = file_name + "_{}.pth".format(i)
                save_path = osp.join(save_dir, save_name)
                print("thread_worker@rank{} | save to: {}".format(rank, save_path))
                # save pth to disk
                if not osp.exists(save_path):
                    torch.save(out[i][None, ...].cpu(), save_path)
                
                # append meta info
                self.out_results[rank].append(
                    dict(
                        ori_img_name=file_name,
                        file_name=save_name,
                        category_name=category_name,
                        attributs_name=attributes_name,
                        bbox=bbox,
                        area=area
                    )
                )

            # save to disk
            # for i in range(len(paths)):
            #     path = paths[i] + ".pth"
            #     torch.save(out[i][None, ...].cpu(), path)

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
        
        self.annotation_file = osp.join(annotation_dir, "{}.json".format(self.split))
        print("annotation will be cached at: ", self.annotation_file)

    def run(self):
        if self.mp_size == 1:
            self.thread_worker(0)
        else:
            processes = []
            for rank in range(self.mp_size):
                p = mp.Process(target=self.thread_worker, args=(rank,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        
        # merge results for each rank:
        results = []
        for rank in range(self.mp_size):
            results.extend(self.out_results[rank])
        with open(self.annotation_file, "w") as f:
            f.write(json.dumps(results, indent=2))
        print("cache finished, annotation file saved at: ", self.annotation_file)
        