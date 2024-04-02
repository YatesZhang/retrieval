import torch 
from torch.utils.data import Dataset, DataLoader
import random 
import cv2 
import pdb 
import numpy as np
from PIL import Image 
from copy import deepcopy 
from torchvision import datasets
from tqdm import tqdm
import sys 
sys.path.append('..')

from Flamingo.models.modeling_sam import SAMImageTransforms
from prompter import COCOPrompter

def get_random_color():
    """ 
        random_color: [1, 1, 3]
        mask[:, :, None]: [H, W, 1]
        random_color * mask
    """
    color = random.randint(0,255), random.randint(0,255), random.randint(0,255)
    # shape: [1, 1, 3]
    return np.array(color)[None, None, :].astype('uint8')

class PromptCOCO(Dataset):
    def __init__(self, cocodetection, sam_transforms=None):
        """ 
            cocodetection: 
        """
        self.dataset = cocodetection  
        self.sam_transforms = sam_transforms
        
    def __len__(self):
        return len(self.dataset)    

    def generate_masks(self, target, img_size):
        """ 
            generate masks from target for semanitc segmentation

            return:
                category_ids_new: [int]
                masks_new: [mask: (H,W)]
        """

        category_set = set([ann['category_id'] for ann in target])
        masks = dict() 
        for category_id in category_set:
            masks[category_id] = np.zeros(img_size)

        for ann in target:
            category_id = ann['category_id']
            mask = self.dataset.coco.annToMask(ann)
            masks[category_id] += mask  
        
        masks_new = []
        category_ids_new = []
        for category_id in masks:
            category_ids_new.append(category_id)
            mask = (masks[category_id] > 0).astype("uint8")
            masks_new.append(mask)

        return category_ids_new, masks_new 
        

    def __getitem__(self, idx):
        """ 
            image: PIL.Image.Image -> np.ndarray

            target: [
                {
                    'segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'
                }
            ]
        """
        image, target = deepcopy(self.dataset[idx])
        image = np.array(image)
        ori_img_size = image.shape[:2] 

        # generate masks:
        category_ids, masks = self.generate_masks(target, ori_img_size)

        # apply sam_transforms:
        if self.sam_transforms is not None:
            """ 
                masks: [mask: (H,W)]
                image: (1, C, H, W)
            """
            # import pdb 
            # pdb.set_trace()
            image, masks = self.sam_transforms(image=image, mask=masks)
            if isinstance(masks, list):
                masks = torch.cat(masks, dim=0)
                assert masks.dim() == 3, "masks should be [N_{categories}, H, W] after sam_transforms"
        return dict(
            ori_img_size=ori_img_size,
            image=image,
            category_ids=category_ids,
            masks=masks,
        )


def collate_fn_coco(data: list):
    """ 
        merge data dict list to one dict
        return:
            dict(
                images: (B, C, H, W)
                masks: List[(N_{categories}, H, W)]   len = B
                ori_img_sizes: List[(h, w)]           len = B
                category_ids: List[List[catid]]       len = B                    
            )
    """
    images = torch.cat([d['image'] for d in data], dim=0)
    masks = [d['masks'] for d in data]
    ori_img_sizes = [d['ori_img_size'] for d in data]
    category_ids = [d['category_ids'] for d in data]
    return dict(
        images=images,
        masks=masks,
        ori_img_sizes=ori_img_sizes,
        category_ids=category_ids,
    )

def build_COCO(img_dir, anno_file, sam_transforms=None):
    """ 
        img_dir: coco image path
        anno_file: coco annotation file path
    """
    dataset = datasets.CocoDetection(img_dir, anno_file)
    return PromptCOCO(dataset, sam_transforms=sam_transforms)


class DataIterator:
    def __init__(self, dataloader: PromptCOCO, prompter, cat_padding=4):
        """ 
            pad to 4 categories for each image
        """
        self.cat_padding = cat_padding 
        self.dataloader = dataloader 
        self.prompter = prompter 
        self.iter = iter(dataloader)
    
    def local_padding(self, category_ids, prompts):
        """ 
        use local prompt to pad category_ids
            category_ids: List[List[catid]]
            prompts: List[prompt]
                - prompt:
                    dict(
                        catIds: List[catid],
                        catNames: List[category_name],
                        catMask: List[category_mask],
                        instances=List[torch_instance]
                    )
        """
        category_ids_new = []
        catMask_new = []
        instances_new = []
        for i in range(len(category_ids)):
            # pad prompt i
            category_id = category_ids[i].copy()
            instance = prompts[i]['instances'].clone()
            catMask = prompts[i]['catMask'].copy()
            while len(category_id) < self.cat_padding:
                # search prompts[j] (j != i): 
                for j in range(len(prompts)):
                    if i == j: continue
                    prompt = prompts[j]
                    catIds = prompt['catIds']    # catIds: List[catid]
                    for k, catId in enumerate(catIds):
                        if catId not in category_id:
                            # do padding:
                            category_id.append(catId)
                            catMask.append(0)
                            # instance: from prompts_{i}
                            # prompt:   from prompts_{j}
                            instance = torch.cat([instance, prompt['instances'][k][None, :, :]], dim=0)
                        if len(category_id) >= self.cat_padding: break
                    if len(category_id) >= self.cat_padding: break
                if len(category_id) < self.cat_padding:
                    # no more catId to append, accept:
                    # TODO: retrieve
                    break 

            category_ids_new.append(category_id)
            catMask_new.append(catMask)
            instances_new.append(instance)
        # return category_ids
                        


                

        return category_idsr

    def __iter__(self):
        return self 

    def __next__(self):
        """ 
            data: 
                dict(
                    images: (B, C, H, W)                  torch
                    masks: List[(N_{categories}, H, W)]   len = B
                    ori_img_sizes: List[(h, w)]           len = B
                    category_ids: List[List[catid]]       len = B                    
                )
            
            prompt:
                dict(
                    catIds: List[catid],
                    catNames: List[category_name],
                    catMask: List[category_mask],
                    instances=List[torch_instance]
                )
        """
        try:
            # unpack data:
            data = next(self.iter)
            images = data['images']
            masks = data['masks']
            ori_img_sizes = data['ori_img_sizes']
            category_ids = data['category_ids']
            
            # check data
            B = len(category_ids)
            assert B == len(masks) and B = images.shape[0] and B == len(ori_img_sizes)

            prompts = []
            for i in range(B):
                category_id = category_ids[i]

                # retrieve prompt:
                prompt = self.prompter[category_id]
                prompts.append(prompt)
            
                # if len(category_id) < self.cat_padding:
                #     self.random_padding()
                # catIds = prompt['catIds']
                # catNames = prompt['catNames']
                # catMask = prompt['catMask']
                # instances = prompt['instances']



        except StopIteration:
            self.iter = iter(self.dataloader)
            raise StopIteration


    # def generate_
if __name__ == '__main__':
    ROOT = "/root/datasets/COCO" 
    IMAGES_PATH = "/root/datasets/COCO/train2017"
    ANNOTATIONS_PATH = "/root/datasets/COCO/annotations/instances_train2017.json" 
    # ROOT = "/root/datasets/OpenDataLab___PASCAL_VOC2012/format2COCO" 
    # IMAGES_PATH = "/root/datasets/OpenDataLab___PASCAL_VOC2012/format2COCO/val"
    # ANNOTATIONS_PATH = "/root/datasets/OpenDataLab___PASCAL_VOC2012/format2COCO/annotations/val.json"
    
    
    transforms = SAMImageTransforms(long_side_length=1024)
    dataset = build_COCO(IMAGES_PATH, ANNOTATIONS_PATH, sam_transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_coco)
    i = 0
    for epoch in range(50):
        for data in tqdm(dataloader):
            i += 1
            if i >= 100:
                break 
            print(data['images'].shape)
            print([mask.shape for mask in data['masks']])
            print(data['category_ids'])
            print(data['ori_img_sizes'])
            pdb.set_trace()
            print()
        if i >= 100:
            break 
