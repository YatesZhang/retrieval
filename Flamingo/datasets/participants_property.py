
import json
import pdb 
from torch.utils.data import Dataset
import cv2
import os

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
            img=np.ndarray(1080, 1920, 3)
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
    
    def load_image(self, data_info):
        file_name = data_info['file_name']
        path = os.path.join(self.imgs_dir, file_name)
        img = cv2.imread(path)
        data_info['img'] = img
        return data_info

    def draw_labels_with_data_info(self, data_info):
        img = data_info['img'].copy()
        annotations = data_info['annotations']
        for item in annotations:
            bbox = item['bbox']
            category_name = item['category_name']
            attributes_name = item['attributes_name']
            text = "{}: {}".format(category_name, attributes_name)
            img = draw_bounding_box(img, bbox, text, mode='xywh')
        return img

    def __getitem__(self, index):
        data_info = self.data_infos[index].copy()
        data_info = self.load_image(data_info)
        # data_info['img'] = self.draw_labels_with_data_info(data_info)
        return data_info
        

# if __name__ == '__main__':
#     annFile = "/root/datasets/participant_property/participant_property/labels/val/valid_coco.json"
#     imgs_dir = "/root/datasets/participant_property/participant_property/images"
#     dataset = ParticipantsProperty(annFile=annFile, imgs_dir=imgs_dir)
#     for data in dataset:
#         print(data)
#         cv2.imwrite("./test.jpg", data['img'])
#         pdb.set_trace()