from torch.utils.data import Dataset, DataLoader
import random 
import cv2 
import numpy as np
from PIL import Image 
from copy import deepcopy 
from torchvision import datasets
from tqdm import tqdm

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
            image, masks = self.sam_transforms(image=image, mask=masks)
        return dict(
            ori_img_size=ori_img_size,
            image=image,
            category_ids=category_ids,
            masks=masks,
        )


def collate_fn_coco(data: list):
    """ 
        merge data dict list to one dict
    """
    pass 

def build_COCO(img_dir, anno_file, sam_transforms=None):
    """ 
        img_dir: coco image path
        anno_file: coco annotation file path
    """
    dataset = datasets.CocoDetection(img_dir, anno_file)
    return PromptCOCO(dataset, sam_transforms=sam_transforms)

if __name__ == '__main__':
    ROOT = "/root/datasets/COCO" 
    IMAGES_PATH = "/root/datasets/COCO/train2017"
    ANNOTATIONS_PATH = "/root/datasets/COCO/annotations/instances_train2017.json" 
    # ROOT = "/root/datasets/OpenDataLab___PASCAL_VOC2012/format2COCO" 
    # IMAGES_PATH = "/root/datasets/OpenDataLab___PASCAL_VOC2012/format2COCO/val"
    # ANNOTATIONS_PATH = "/root/datasets/OpenDataLab___PASCAL_VOC2012/format2COCO/annotations/val.json"
    dataset = build_COCO(IMAGES_PATH, ANNOTATIONS_PATH, sam_transforms=None)
    i = 0
    for epoch in range(50):
        for data in tqdm(dataset):
            i += 1
            if i >= 9000:
                break 
        if i >= 9000:
            break 
