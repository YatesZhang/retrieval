""" 
    Detection class
"""
from PIL import Image
def crop(img, bbox):
    """
        crop a pillow image
    """
    assert isinstance(img, Image.Image)
    x, y, w, h = bbox
    if x <=0 or y <=0 or w<=0 or h<=0:
        raise ValueError('bbox is not valid')
    x, y, w, h = int(x), int(y), int(w), int(h)
    return img.crop((x, y, x + w, y + h))


class Detection2CLSLabel(object):
    """ 
    data_info: 
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
    def __init__(self, data_info):
        """
            accept data info from dataset
            Flamingo.datasets.participants_property
        """
        # abspath
        img_path = data_info['path']
        # name
        file_name = data_info['file_name']
        # PIL image
        img = data_info['img']
        # annotations
        annotations = data_info['annotations']
        
        # crop image from bbox:
        self.imgs = []
        self.category_names = []
        self.attributes_names = []
        self.metas = []

        for annotation in annotations:
            instance_id = annotation['instance_id']
            category_name = annotation['category_name']
            attributes_name = annotation['attributes_name']
            bbox = annotation['bbox']
            area = annotation['area']

            # skip invalid bbox:
            x, y, w, h = bbox
            if x <=0 or y <=0 or w<=0 or h<=0:
                continue
            if  x + w >= img.width or y + h >= img.height:
                continue
            self.imgs.append(crop(img, bbox))
            self.category_names.append(category_name)
            self.attributes_names.append(attributes_name)
            self.metas.append(
                dict(
                    instance_id=instance_id,
                    area=area,
                    bbox=bbox,
                    img_path=img_path,
                    file_name=file_name,
                )
            )
        pass 
    
    def __len__(self):
        """ 
            lenght of the detection result
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """ 
            meta: 
                dict(
                    instance_id=...,
                    area=...,
                    bbox=...,
                    img_path=...,
                    file_name=...
                )
        """
        return dict(
            img=self.imgs[idx],
            category_name=self.category_names[idx],
            attributes_name=self.attributes_names[idx],
            meta=self.metas[idx],
        )

def merge_detected(batch):
    """

    """
    imgs = []
    category_names = []
    attributes_names = []
    metas = []
    for detection in batch:
        assert isinstance(detection, Detection2CLSLabel)
        imgs += detection.imgs    # List extend
        category_names += detection.category_names
        attributes_names += detection.attributes_names
        metas += detection.metas
    return dict(
        imgs=imgs,
        category_names=category_names,
        attributes_names=attributes_names,
        metas=metas,
    )