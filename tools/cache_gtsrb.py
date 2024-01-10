"""
    CUDA_VISIBLE_DEVICES=2 python cache_gtsrb.py
"""

import os 
from rich import print 
import torch 
from torch.utils.data import Dataset, DataLoader 

from PIL import Image 
from glob import glob
import numpy as np 

# Resizing the images to 30x30x3
IMG_HEIGHT = 224
IMG_WIDTH = 224
channels = 3
# NUM_CATEGORIES = len(os.listdir(train_path))
# print(NUM_CATEGORIES)

# Label Overview
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

class GTSRB(Dataset):
    def __init__(self,
                data_dir='/home/yunzhi/yunzhi/datasets',
                train_path='/root/datasets/GTSRB/Train',
                test_path='/home/yunzhi/yunzhi/datasets',
                mode='train') -> None:
        super().__init__()
        self.data_dir = data_dir,
        self.train_path = train_path
        self.test_path = test_path
        self.data_infos = []
        print("load annos:")
        self.load_annotations()

    def show(self, index):
        data_info = self.data_infos[index].copy()
        path = data_info['path']
        label = data_info['label']
        print(label)
        return Image.open(path)

    def load_annotations(self):
        # load Annotation
        self.data_infos = []
        for cat in classes:
            label = classes[cat]
            img_dir = os.path.join(self.train_path, str(cat))
            img_names = glob(os.path.join(img_dir, "*.png"))
            # print(img_dir, img_names)
            for name in img_names:
                data_info = dict(
                    path=os.path.join(img_dir, name),
                    label=label
                )
                self.data_infos.append(data_info)
        return 

    def __getitem__(self, index):
        data_info = self.data_infos[index].copy()
        path = data_info['path']
        label = data_info['label']
        return dict(
            img=Image.open(path),
            label=label,
            path=path
        )
    
    def __len__(self):
        return len(self.data_infos)


try:
    import Flamingo
except ModuleNotFoundError:
    import sys 
    sys.path.append("..")

from Flamingo.models.batchprocessor import CLIPBatchProcessor
from Flamingo.lora_tuning import get_tokenizer
from Flamingo.models.modeling_clip import get_clip_vision_encoder_and_processor
from tqdm import tqdm 

def collate_fn(batch):
    imgs = []
    # labels = []
    paths = []
    for data in batch:
        imgs.append(data['img'])
        paths.append(data['path'])
        # labels.append(data['label'])
    return dict(
        path=paths,
        img=imgs
    )
if __name__ == '__main__':
    vision_encoder, image_processor = get_clip_vision_encoder_and_processor()
    vision_encoder = vision_encoder.cuda()
    batch_processor = CLIPBatchProcessor(vision_encoder=vision_encoder,
                                        image_processor=image_processor)
    dataset = GTSRB()
    dataloader = DataLoader(dataset=dataset,
                            batch_size=24, shuffle=False,
                            num_workers=12, collate_fn=collate_fn)

    for data in tqdm(dataloader):
        imgs = data['img']
        paths = data['path']
        out = batch_processor(imgs)    # [B, 1, 1, V, D]
        
        # save to disk
        for i in range(len(paths)):
            path = paths[i] + ".pth"
            torch.save(out[i][None, ...].cpu(), path)