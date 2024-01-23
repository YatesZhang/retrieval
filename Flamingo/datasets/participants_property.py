
import json
import pdb 
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import os
from copy import deepcopy
try:
    import Flamingo
except ModuleNotFoundError:
    import sys
    sys.path.append("../..")
    import Flamingo
from Flamingo.structure import Detection2CLSLabel
from glob import glob


def draw_bounding_box(image, bbox, label, mode='xyxy'):
    if mode == 'xywh':
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    elif mode == 'xyxy':
        x1, y1, x2, y2 = bbox
        x, y = x1, y1
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = "{}".format(label)
    cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 4)
    return image

# def draw_bounding_boxes(image, bboxes, label):
#     for bbox in bboxes:
#         image = draw_bounding_box(image, bbox)
#     return image

    
class ParticipantsProperty(Dataset):
    """ 
    getitem: 
        dict(
            path=...,
            file_name=...,
            annotations=[
                dict(
                    instance_id=...,
                    category_name=...,
                    attributes_name=...,
                    bbox=[x, y, w, h],
                    area=...
                )
            ],
            img=Image(1080, 1920, 3)
        )
    """
    def __init__(self, annFile, imgs_dir):
        super(ParticipantsProperty, self).__init__()
        self.attributes = {1:'not obstructed',
                            2:'0-50 obstruction',
                            3:'50-80 obstruction'}
        self.imgs_dir = imgs_dir
        # load annotations:
        annos = None
        with open(annFile, "r") as f:
            annos = json.load(f)
        _images = annos['images']
        annotations = annos['annotations']
        categories = annos['categories']

        # step 1) index categories:
        self.cats = {}
        for item in categories:
            self.cats[item['id']] = item['name'].replace('_', ' ')
        
        # step 2) index images:
        images = {}
        for item in _images:
            images[item['id']] = dict(
                file_name=item['file_name'],
                path=os.path.join(self.imgs_dir, item['file_name']),
                annotations=[]
            )
        
        # step 3) index annotations:
        for item in annotations:
            instance_id = item['id']
            image_id = item['image_id']
            category_id = item['category_id']
            bbox = item['bbox']
            area = item['area']
            # pdb.set_trace()
            category_name = self.cats[category_id[0]]
            attributes_name = self.attributes[category_id[1]]
            images[image_id]['annotations'].append(dict(
                instance_id=instance_id,
                category_name=category_name,
                attributes_name=attributes_name,
                bbox=bbox,
                area=area
            ))

        # step 4) convert to list:
        self.data_infos = []
        for k in images:
            self.data_infos.append(images[k])

    def __repr__(self):
        return "ParticipantsProperty(num_imgs={}, \n  categories={})".format(
            len(self.data_infos),
            str(self.cats)
        )

    def __len__(self):
        return len(self.data_infos)
    
    def split(self, mp_size):
        """ 
           mp_size: muti-threading size
        """
        datasets = []
        for i in range(mp_size):
            dataset = deepcopy(self)
            dataset.data_infos = self.data_infos[i::mp_size]
            datasets.append(dataset)
        assert sum([len(dataset) for dataset in datasets]) == len(self.data_infos)
        return datasets

    def load_image(self, data_info):
        file_name = data_info['file_name']
        path = os.path.join(self.imgs_dir, file_name)
        img = Image.open(path)
        data_info['img'] = img
        return data_info

    def draw_labels_with_data_info(self, data_info):
        img = data_info['img'].copy()
        if isinstance(img, Image.Image):
            img = np.array(img)
        annotations = data_info['annotations']
        for item in annotations:
            bbox = item['bbox']
            category_name = item['category_name']
            attributes_name = item['attributes_name']
            text = "{}: {}".format(category_name, attributes_name)
            img = draw_bounding_box(img, bbox, text, mode='xywh')
        return img
    def collate_fn(self, batch):
        batch = [Detection2CLSLabel(data_info) for data_info in batch]
        pass 
    def __getitem__(self, index):
        data_info = self.data_infos[index].copy()
        data_info = self.load_image(data_info)
        # data_info['img'] = self.draw_labels_with_data_info(data_info)
        return data_info
        

""" 
    anno_file: train.json
"""
class CachedParticipants(Dataset):
    def __init__(self, data_dir, anno_file, tokenizer):
        """ 
            cache_dir:
            |-- annotations
            |   |-- train.json
            |   |-- val.json
            |-- pth
            |   |-- category_name
            |       |-- *.pth
        """
        self.data_dir = data_dir
        assert data_dir.endswith('pth')

        self.anno_file = anno_file
        self.tokenizer = tokenizer
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        """ 
            annotations:
            List[
                dict(
                    ori_img_name,
                    file_name,
                    category_name,
                    attributes_name,
                    bbox=[x, y, w, h],
                    area,
                )
            ]
        """
        # set up labels; language model is expected to handle shifting
        # encode media token id will add a token id of 2 before special token
        self.media_token_id = self.tokenizer.encode("<image>")[-1]

    def collater(self, samples):
        """
            sample: 
                dict(
                    vision_x=vision_x,
                    category_name=category_name,
                    attributes_name=attributes_name,
                    # meta :
                    ori_img_name=ori_img_name,
                    file_name=file_name,
                    bbox=bbox,
                    area=area,
                    pth_file=pth_file
                )

        """

        imgs = []
        text_labels = [] 
        paths = []
        input_ids = []
        attention_mask = []
        metas = []
        for sample in samples:
            # get vision label:
            img = sample['vision_x']
            assert len(img.shape) == 5, "img should be shape of [B, T, F, V, D] and T==1, F==1"
            imgs.append(img)

            # get text label:
            category_name = sample['category_name']
            attributes_name = sample['attributes_name']
            label = attributes_name + " " + category_name
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

            # get meta info:
            pth_file = sample['pth_file']
            ori_img_name = sample['ori_img_name']
            bbox = sample['bbox']
            meta = dict(
                pth_file=pth_file,
                ori_img_name=ori_img_name,
                bbox=bbox,
            )
            metas.append(meta)

            # max_length padding
            """ 
            max_length = 20 padding:
                </s><image>Output:motorcycle electric vehicle 50-80 obstruction<|endofchunk|>\
                    </s><pad><pad><pad><pad><pad>
            """
            batch_encoding = self.tokenizer(label,
                                            max_length=20,
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
            meta=metas,
            vision_x=imgs, 
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels 
        )
        return result

    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        """ 
            unpack the annotation dict
        """
        data_info = self.annotations[index].copy()
        ori_img_name = data_info['ori_img_name']
        file_name = data_info['file_name']
        category_name = data_info['category_name']
        if 'attributes_name' in data_info:
            attributes_name = data_info['attributes_name']
        else:
            attributes_name = data_info['attributs_name']
        bbox = data_info['bbox']
        x, y, w, h = bbox
        area = data_info['area']
        
        # get file name
        category_dir = category_name.replace(' ', '_')
        pth_file = os.path.join(self.data_dir, category_dir, file_name)
        # get vision_x
        vision_x = torch.load(pth_file)
        return dict(
            vision_x=vision_x,
            category_name=category_name,
            attributes_name=attributes_name,
            # meta :
            ori_img_name=ori_img_name,
            file_name=file_name,
            bbox=bbox,
            area=area,
            pth_file=pth_file
        )

if __name__ == '__main__':
    annFile = "/root/datasets/participant_property/participant_property/labels/val/valid_coco.json"
    imgs_dir = "/root/datasets/participant_property/participant_property/images"
    dataset = ParticipantsProperty(annFile=annFile, imgs_dir=imgs_dir)
    for data in dataset:
        print(data)
        # cv2.imwrite("./test.jpg", data['img'])
        pdb.set_trace()