import open_clip
from PIL import Image 
import numpy as np 
import pdb 
import torch 

""" 
    on .13:
    "/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo"
"""
def get_clip_vision_encoder_and_processor(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    cache_dir = None   
):
    """
        get clip vision encoder
        return vision_encoder.visual and image processor
    """
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path,    # "ViT-L-14"
        pretrained=clip_vision_encoder_pretrained,    # "openai"
        cache_dir=cache_dir,
    )
    vision_encoder.visual.output_tokens = True
    encoder = vision_encoder.visual 
    return encoder, image_processor 

# if __name__ == "__main__":
#     clip_vision_encoder_path="ViT-L-14"
#     clip_vision_encoder_pretrained="openai"
#     cache_dir = "/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo"
#     vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
#         clip_vision_encoder_path,    # "ViT-L-14"
#         pretrained=clip_vision_encoder_pretrained,    # "openai"
#         cache_dir=cache_dir,
#     )
#     # set the vision encoder to output the visual features
#     vision_encoder.visual.output_tokens = True
    
#     img = Image.open("/home/yunzhi/yunzhi/datasets/meta/1.png")
#     # input: Pillow format image
#     img = image_processor(img)
#     img = img[None, ...].cuda()    # [1, C, H, W]
#     encoder = vision_encoder.visual 
#     encoder = encoder.cuda()
#     encoder.eval()
#     """ 
#     encoder:
#         encoder(img)[0].shape: torch.Size([1, 768])
#         encoder(img)[1].shape: torch.Size([1, 256, 1024])
#     """
#     with torch.no_grad():
#         out = encoder(img)[1]
#         out = out.unsqueeze(1).unsqueeze(1)
#         print("(B, T, F) v, d | output[1] from CLIP.visual_encoder.vision", out.shape)
#     pdb.set_trace()
    