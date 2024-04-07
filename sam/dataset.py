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
import open_clip 

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
        self.coco = cocodetection.coco
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
        bug:
            target should not be empty!
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
                try:
                    masks = torch.cat(masks, dim=0) if len(masks) >0 else torch.empty((0, *image.shape[-2:]))
                except RuntimeError:
                    import pdb 
                    pdb.set_trace()
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
    def __init__(self, dataloader, prompter, cat_padding=4):
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
            category_ids: List[catid]
            prompts: List[prompt]
                - prompt:
                    dict(
                        catIds: List[catid],
                        catNames: List[category_name],
                        catMask: List[category_mask],
                        instances=List[torch_instance]
                    )
        """

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
            assert B == len(masks) and B == images.shape[0] and B == len(ori_img_sizes)

            for i in range(B):
                masks[i] = masks[i][:self.cat_padding]
                category_ids[i] = category_ids[i][:self.cat_padding]
                ori_img_sizes[i] = ori_img_sizes[i][:self.cat_padding]

            prompts = []
            for i in range(B):
                # retrieve prompt:
                category_id = category_ids[i]
                prompt = self.prompter[category_id]
                prompts.append(prompt)
            
            for i in range(B - 1):
                prompt_i = prompts[i]
                for j in range(i+1, B):
                    prompt_j = prompts[j]
                    # merge prompt_i and prompt_j
                    catIds_i = prompt_i['catIds']
                    catNames_i = prompt_i['catNames']
                    catMask_i = prompt_i['catMask']
                    instances_i = prompt_i['instances']

                    catIds_j = prompt_j['catIds']
                    catNames_j = prompt_j['catNames']
                    catMask_j = prompt_j['catMask']
                    instances_j = prompt_j['instances']
                    if len(catIds_i) >= self.cat_padding and len(catIds_j) >= self.cat_padding:
                        continue 
                    if len(catIds_j) < self.cat_padding:
                        # pad cat j
                        for k in range(len(catIds_i)):
                            catId_i = catIds_i[k]
                            if catId_i not in catIds_j:
                                # merge cat id:
                                catIds_j.append(catId_i)
                                # merge cat name:
                                catNames_j.append(catNames_i[k])
                                # add cat mask:
                                catMask_j.append(0)
                                instances_j.append(instances_i[k])
                                masks[j] = torch.cat([masks[j], torch.zeros((1, *masks[j].shape[-2:]))], dim=0)
                            if len(catIds_j) >= self.cat_padding:
                                # stop padding
                                break
                    if len(catIds_i) < self.cat_padding:
                        # pad cat i
                        for k in range(len(catIds_j)):
                            catId_j = catIds_j[k]
                            if catId_j not in catIds_i:
                                # merge cat id:
                                catIds_i.append(catId_j)
                                # merge cat name:
                                catNames_i.append(catNames_j[k])
                                # add cat mask:
                                catMask_i.append(0)
                                instances_i.append(instances_j[k])
                                masks[i] = torch.cat([masks[i], torch.zeros((1, *masks[i].shape[-2:]))], dim=0)
                            if len(catIds_i) >= self.cat_padding:
                                # stop padding
                                break
                    prompts[i] = dict(
                        catIds=catIds_i,
                        catNames=catNames_i,
                        catMask=catMask_i,
                        instances=instances_i
                    )
                    prompts[j] = dict(
                        catIds=catIds_j,
                        catNames=catNames_j,
                        catMask=catMask_j,
                        instances=instances_j
                    )
            
            # process mask:
            # import pdb 
            # pdb.set_trace()
            for i in range(B):
                prompt = prompts[i]
                catMask = prompt['catMask']
                for j, cat_mask in enumerate(catMask):
                    if cat_mask == 0:
                        mask_i = masks[i]    # shape: [N_{categories}, H, W]
                        mask_i[j] = 0
                        masks[i] = mask_i    # 防止弱引用
                
            # merge return value:
            labels = [prompt['catIds'] for prompt in prompts]
            names = [prompt['catNames'] for prompt in prompts]
            prompts = [prompt['instances'] for prompt in prompts]

            return dict(
                images=images,
                labels=labels,
                masks=masks,
                names=names,
                prompts=prompts
            )

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
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_coco, num_workers=4)

    print("create image processor for clip")
    _, _, image_processor = open_clip.create_model_and_transforms(
        "ViT-L-14",    # "ViT-L-14"
        pretrained="openai",    # "openai"
        cache_dir=None,
    )

    print("create prompter")
    prompter = COCOPrompter(annFile=ANNOTATIONS_PATH,
                            img_dir=IMAGES_PATH,
                            shot=5,
                            mask_rate=0.25,
                            coco=dataset.coco,
                            transforms=image_processor)

    data_iter = DataIterator(dataloader, prompter, cat_padding=4)
    i = 0
    for data in tqdm(data_iter):
        i += 1
        if i == 10:
            pdb.set_trace()
    # i = 0
    # for epoch in range(50):
    #     for data in tqdm(dataloader):
    #         i += 1
    #         if i >= 100:
    #             break 
    #         print(data['images'].shape)
    #         print([mask.shape for mask in data['masks']])
    #         print(data['category_ids'])
    #         print(data['ori_img_sizes'])
    #         pdb.set_trace()
    #         print()
    #     if i >= 100:
    #         break 
