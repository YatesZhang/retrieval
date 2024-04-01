import cv2 
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from typing import Tuple, List


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int=1024) -> Tuple[int, int]:
    """
    resize image to : 
    upper bound of [1024 * (h/w), 1024]
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


class Normalize(torch.nn.Module):
    def __init__(self) -> None:
        """ 
            parameters can be deployed on GPU
        """
        super().__init__()
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.img_size = 1024

    def pad(self, x):
        # Pad to square:
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        x = self.pad(x)
        return x
    
    def forward(self, batched_input):
        """ 
            input: 
                batched_input: [N, C, H, W]
        """
        batched_input = torch.stack([self.preprocess(x) for x in batched_input], dim=0)
        return batched_input

class SAMImageTransforms(object):
    def __init__(self, long_side_length=1024):    
        self.norm = Normalize()
    
    def process_image(self, image, target_size):
        """ 
            process a single image 
        """
        # process image:
        image = np.array(resize(to_pil_image(image), target_size))
        input_image_torch = torch.as_tensor(image)    # as tensor
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image_torch = self.norm(input_image_torch)    # normalize and padding 
        
        return input_image_torch
    
    def get_mask_shape(self, mask):
        """ 
            get mask shape recurrsively
        """
        if isinstance(mask, np.ndarray):
            return mask.shape[:2]
        elif isinstance(mask, list):
            return self.get_mask_shape(mask[0])
        else:
            raise NotImplementedError
            
    def process_masks(self, masks, target_size=None):
        """ 
            process masks recurrsively
            support type:
            - ndarray
            - list
            - list[list]
        """
        if masks is None:
            return None 
        if target_size is None:
            target_size = get_preprocess_shape(*self.get_mask_shape(masks), long_side_length=1024)
        if isinstance(masks, np.ndarray):
            mask = np.array(resize(to_pil_image(mask), target_size))
            mask = torch.as_tensor(mask)
            mask = mask.contiguous()[None, :, :]
            mask = self.norm.pad(mask)    # only padding
            return mask 
        elif isinstance(masks, list):
            return [self.process_masks(mask, target_size) for mask in masks]
        else:
            raise NotImplementedError

    def __call__(self, image=None, mask=None):
        if image is None:
            return None 
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert isinstance(image, np.ndarray)

        if mask is not None: 
            assert isinstance(mask, np.ndarray)
            assert mask.shape[0] == image.shape[0] and mask.shape[1] == image.shape[1]
            assert len(mask.shape) == 2

        # get target size:
        target_size = get_preprocess_shape(image.shape[0], image.shape[1], long_side_length=1024)
        
        # process image:
        input_image_torch = self.process_image(image, target_size)    # normalize and padding 
        mask = self.process_masks(mask, target_size)

        # process mask:
        if mask is not None:
            return input_image_torch, mask
        else:
            return input_image_torch
            
if __name__ == "__main__":
    image = "/root/yunzhi/flamingo_retrieval/retrieval/Flamingo/images/yellow_bus.jpg"
    transforms = SAMImageTransforms(long_side_length=1024)
    input_image = transforms(image)
