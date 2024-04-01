import pycocotools
from pycocotools.coco import COCO
from copy import deepcopy
import random 
from functools import  lru_cache   
from PIL import Image 
import numpy as np 
import os.path as osp
import pdb 
from tqdm import tqdm 
from torch.utils.data import Dataset

class COCOPrompter(Dataset):
    def __init__(self, annFile, img_dir, shot=5, mask_rate=0.25, coco=None):
        super().__init__()
        if coco is not None:
            self.coco = coco 
        else:
            self.coco = COCO(annFile) 
        self.shot = shot
        self.mask_rate =  mask_rate
        self.img_dir = img_dir
        
        # init start index for each category
        catIds = self.coco.getCatIds()
        self.start_idx = dict()
        for i in catIds:
            self.start_idx[i] = 0

        self.catIds = catIds.copy()
    
    def _len__(self):
        return  len(self.catIds)

    def get_cat_mask(self, catIds):
        """ 
            mask some categories with mask_rate

            return: new catIds with mask 

            e.g.
                cat_mask: [1, 1, 1, 0]
                catIds: [1, 30, 9, 4]
                category 4 is masked
        """     
        masked_num = int(self.mask_rate * len(catIds))
        if masked_num < 1:
            return [1 for _ in range(len(catIds))] 
        else: 
            mask_idx = random.sample(range(len(catIds)), masked_num)  
            mask = [1 for _ in range(len(catIds))]
            for i in mask_idx:
                mask[i] = 0
            return mask 

    def replace_masked_cat_with_other(self, catIds, mask):
        """ 
            mask: see self.get_cat_mask()
            e.g.
                cat_mask: [1, 1, 1, 0]
                catIds: [1, 30, 9, 4]
                category 4 is masked
                4 will be replaced with other category not in [1, 30, 9]
        """
        random.shuffle(self.catIds)
        # count mask:
        mask_num = len(mask) - sum(mask)
        
        if mask_num <= 0:
            # no need to replace
            return catIds 
        else: 
            ans = []
            for cat in self.catIds: 
                if cat not in catIds:
                    ans.append(cat)
                    if len(ans) == mask_num:
                        break
            
            j = 0
            for i in range(len(catIds)):
                if mask[i] == 0: 
                    catIds[i] = ans[j]
                    j += 1 
            return catIds 
            
    @lru_cache(maxsize=50)
    def imread(self, imgId):
        if isinstance(imgId, int):
            imgId = str(imgId) 
        imgId = imgId.zfill(12) + ".jpg"
        imgId = osp.join(self.img_dir, imgId)
        return np.array(Image.open(imgId))

    def get_instances_from_idx(self, idx: list,  catId: int):
        """ 
            idx: imgIds, N-shot images of category catId
            we need to random sample N-shot instances from N-shot images of category catId
            anns: [
                dict(
                    'segmentation',
                    'area',
                    'iscrowd',
                    'image_id',
                    'bbox',
                    'category_id',
                    'id'
                )
            ]
            idx: [int]
        """
        annIds = self.coco.getAnnIds(imgIds=idx, catIds=[catId], iscrowd=None)
        anns = self.coco.loadAnns(annIds) 
        samples_idx = random.sample(range(len(anns)), self.shot) 
        
        anns_new = []
        for i in samples_idx:
            anns_new.append(anns[i])
        idx_new = list(set([ann['image_id'] for ann in anns_new])) 
        
        # load imgs:
        imgs = dict() 
        for imgId in idx_new:
            imgs[imgId] = self.imread(imgId)
        
        # crop imgs:
        instances = []
        for ann in anns_new: 
            imgId = ann['image_id']
            # mask = self.coco.annToMask(ann)  # binary numpy mask
            img = imgs[imgId]    # numpy array

            # masked_instance = img * mask[:, :, None]    # broadcast
            bbox = ann['bbox']
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            masked_instance = img[y:y+h, x:x+w]
            instances.append(masked_instance)
            if len(instances) == self.shot:
                break 
        return instances 
    

    def get_n_shot_idx(self, catIds: list):
        """ 
            get N-shot image path for each category in catIds

            n_shot_idx: type is [[]]
            sample N-shot imgIds from coco.getImgIds(catIds=catIds)
        """
        n_shot_idx = []
        for cat in catIds:
            start_idx = self.start_idx[cat] 
            ImgIds = self.coco.getImgIds(catIds=[cat])

            len_ImgIds = len(ImgIds) 
            end_idx = start_idx + self.shot 

            idx = []
            if end_idx >= len_ImgIds:
                for i in range(self.shot):
                    idx.append(ImgIds[(start_idx + i) % len_ImgIds])
            else: 
                idx = ImgIds[start_idx: end_idx]
            
            
            n_shot_idx.append(idx)

            # update start index:
            self.start_idx[cat] += 1
            self.start_idx[cat] = self.start_idx[cat] % len_ImgIds
        
        return n_shot_idx 

    def prompt(self, catIds: list, masked=True):
        # get unique category ids:
        catIds = list(set(catIds))
        index = self.coco.getImgIds(catIds=catIds)

        if masked:
            # get mask_rate number of images prompt that not in catIds:
            catMask = self.get_cat_mask(catIds)
        else:
            catMask = [1 for _ in range(len(catIds))]
        # replace masked category with other categories:
        catIds = self.replace_masked_cat_with_other(catIds, catMask)

        n_shot_idx = self.get_n_shot_idx(catIds)
        
        instances = []
        for catId, idx in zip(catIds, n_shot_idx):
            n_shot_instances = self.get_instances_from_idx(idx, catId)
            instances.append(n_shot_instances)
        return dict(
            catIds=catIds,
            catNames=self.coco.loadCats(catIds),
            catMask=catMask,
            instances=instances
        )
    
    def __getitem__(self, idx):
        return self.prompt(catIds=idx)

# prompter = COCOPrompter(annFile, img_dir=IMG_ROOT)

ROOT = "/root/datasets/COCO" 
IMAGES_PATH = "/root/datasets/COCO/train2017"
ANNOTATIONS_PATH = "/root/datasets/COCO/annotations/instances_train2017.json" 

if __name__ == "__main__":
    shot = 5
    prompter = COCOPrompter(ANNOTATIONS_PATH, img_dir=IMAGES_PATH, shot=5, mask_rate=0.25)

    catIDs = [14, 89]
    i = 0
    for i in tqdm(range(1000)):
        # prompt data for catIds:
        data = prompter.prompt(catIDs, masked=True)
        prompt_data = prompter.prompt(catIDs, masked=True)
        if i % 10 == 0:
            print(prompter.start_idx)

    