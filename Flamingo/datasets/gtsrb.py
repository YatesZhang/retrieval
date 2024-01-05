import os 
from rich import print 
import torch 
from torch.utils.data import Dataset, DataLoader 

from PIL import Image 
from glob import glob
import numpy as np 

data_dir = '/home/yunzhi/yunzhi/datasets'
train_path = '/home/yunzhi/yunzhi/datasets/Train'
test_path = '/home/yunzhi/yunzhi/datasets'


# Resizing the images to 30x30x3
IMG_HEIGHT = 224
IMG_WIDTH = 224
channels = 3
NUM_CATEGORIES = len(os.listdir(train_path))

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
class GTSRB(Dataset):
    def __init__(self,
                data_dir='/home/yunzhi/yunzhi/datasets',
                train_path='/home/yunzhi/yunzhi/datasets/Train',
                test_path='/home/yunzhi/yunzhi/datasets',
                tokenizer=None,
                image_processor=None,
                mode='train') -> None:
        super().__init__()
        self.data_dir = data_dir,
        self.train_path = train_path
        self.test_path = test_path
        self.data_infos = []
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        # print("load annos:")
        self.load_annotations()

        # set up labels; language model is expected to handle shifting
        # encode media token id will add a token id of 2 before special token
        self.media_token_id = self.tokenizer.encode("<image>")[-1]
    

    def show(self, index):
        data_info = self.data_infos[index].copy()
        path = data_info['path']
        label = data_info['label']
        print(label)
        return Image.open(path)

    def collater(self, samples):
        """
            sample: 
                dict(img, label, path, batch_encoding)
        """

        imgs = []
        text_labels = [] 
        paths = []
        input_ids = []
        attention_mask = []
        for sample in samples:
            img = sample['img']
            assert len(img.shape) == 5, "img should be shape of [B, T, F, V, D] and T==1, F==1"
            imgs.append(img)
            label = sample['label']
            """
                #TODO:
                1) add prompt template (include special tokens) and eos 
                2) remove padding tokens from loss

                openflamingon remove loss before <image> token
                
                #TODO: add prompt training:
                following this code will remove loss from <|endofchunk|> to <image>
                    endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
                    for endofchunk_idx in endofchunk_idxs:
                        token_idx = endofchunk_idx + 1
                        while (
                            token_idx < labels.shape[1]
                            and labels[i][token_idx] != media_token_id
                        ):
                            labels[i][token_idx] = -100
                            token_idx += 1
            """

            # add text template
            label = f"<image>Output:{label}<|endofchunk|>{self.tokenizer.eos_token}"
            text_labels.append(label)
            paths.append(sample['path'])
            batch_encoding = self.tokenizer(label,
                                            max_length=32,
                                            padding="max_length",
                                            truncation=True,
                                            return_tensors="pt")
            input_ids.append(batch_encoding['input_ids'])
            attention_mask.append(batch_encoding['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        assert len(input_ids.shape) == 2 and len(attention_mask.shape) == 2

        """ 
            generate labels:
                1) hugginface transformer will handle shift logits loss
                2) -100 means ignore in huggingface transformer model
        """
        labels = input_ids.clone()
        # remove loss of media token and pad token
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[labels == self.media_token_id] = -100

        imgs = torch.cat(imgs, dim=0)
        result = dict(
            meta=dict(
                text_labels=text_labels,
                paths=paths
            ),
            vision_x=imgs, 
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels 
        )
        return result
    
    def load_annotations(self):
        # load Annotation
        self.data_infos = []
        for cat in classes:
            label = classes[cat]
            img_dir = os.path.join(self.train_path, str(cat))
            img_names = glob(os.path.join(img_dir, "*.png.pth"))
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
        if path.endswith(".pth"):
            imread = torch.load
        elif path.endswith(".png") or path.endswith(".jpg"):
            imread = Image.open 
        else:
            raise RuntimeError
        img = imread(path)
        if isinstance(img, Image.Image) and self.image_processor is not None:
            img = self.image_processor(img)
        return dict(
            img=img,
            label=label,
            path=path,
        )
    
    def __len__(self):
        return len(self.data_infos)