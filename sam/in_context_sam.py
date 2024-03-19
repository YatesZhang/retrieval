"""
    segment anything in-context
"""

from segment_anything import sam_model_registry
import torch 
from torch import nn
import torch.nn.functional as F 
import open_clip 
import pdb 
from typing import Tuple


class InContextSAM(nn.Module):
    def __init__(
        self, 
        prompt_encoder: nn.Module, 
        sam_model: nn.Module,
        input_size = (1024, 1024),
        original_size = (1080, 1920)
    ):
        super().__init__()
        """ 
            embed_dim: 
            image_embedding_size: 
        """
        self.input_size = input_size 
        self.original_size = original_size 
        self.embed_dim = 256
        self.image_embedding_size = (64, 64)
        self.mask_threshold = 0.0
        """
            pe_layer: positional encoding layer 
            no_mask_embed: no mask prompt 
        """
        self.pe_layer = sam_model.prompt_encoder.pe_layer
        self.no_mask_embed = sam_model.prompt_encoder.no_mask_embed 

        # alignment:
        self.prompt_encoder = nn.Sequential(
            prompt_encoder,
            nn.Linear(prompt_encoder.output_dim, self.embed_dim)
            )
        self.image_encoder = sam_model.image_encoder
        self.mask_decoder = sam_model.mask_decoder
        
    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def forward(self, few_shot_prompt, batch_images, return_logits=True):
        """
            K-way N-shot prompt: [K, N, C, H, W]
            target_image: [B, C, H, W]
        """

        # [B, 3, 1024, 1024] -> [B, 256, 256, 64]
        K, N, C, H, W = few_shot_prompt.shape
        few_shot_prompt = few_shot_prompt.view(K * N, C, H, W)
        batch_features = self.image_encoder(batch_images)

        sparse_embeddings = self.prompt_encoder(few_shot_prompt)    # [K * N, 256]
        sparse_embeddings = sparse_embeddings.reshape((K, N, self.embed_dim))    # [K, N, 256]
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(    
                K, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )    # [K * N, 256, 64, 64]
        # pdb.set_trace()
        # Predict masks
        low_res_masks = []
        iou_predictions = []
        for features in batch_features:
            features = features.unsqueeze(0)
            low_res_mask, iou_prediction = self.mask_decoder(
                image_embeddings=features,
                image_pe=self.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # low_res_mask: [K, 1, 256, 256] -> [1, K, 256, 256]
            # iou_prediction: [1, K, 1]
            low_res_mask = low_res_mask.squeeze(1).unsqueeze(0)
            iou_prediction = iou_prediction.unsqueeze(0)
            # pdb.set_trace()
            low_res_masks.append(low_res_mask)
            iou_predictions.append(iou_prediction)
        low_res_masks = torch.cat(low_res_masks, dim=0) 
        iou_predictions = torch.cat(iou_predictions, dim=0)
        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        if not return_logits:
            masks = masks > self.mask_threshold
        """
            low_res_masks: [B, K, 256, 256]
            masks: [B, K, 1080, 1920]
            iou_predictions: [B, K, 1]
        """
        return masks, iou_predictions, low_res_masks


def create_prompt_encoder(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    cache_dir=None 
):
    print("get vision encoder and image processor")
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path,    # "ViT-L-14"
        pretrained=clip_vision_encoder_pretrained,    # "openai"
        cache_dir=cache_dir,
    )
    # set the vision encoder to output the visual features
    assert vision_encoder.visual.output_tokens == False

    # get VisionTransformer from open_clip:
    encoder = vision_encoder.visual 
    return encoder, image_processor

def create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    clip_cache_dir=None, 
    sam_checkpoint_path=None,
):
    """
        create clip as SAM prompt encoder:
    """
    prompt_encoder, clip_prossesor = create_prompt_encoder(
        clip_vision_encoder_path=clip_vision_encoder_path,
        clip_vision_encoder_pretrained=clip_vision_encoder_pretrained,
        cache_dir=clip_cache_dir
    )

    """ 
        create SAM model:
    """
    backbone = 'vit_b'
    for vit in ['vit_b', 'vit_l', 'vit_h']:
        if vit in sam_checkpoint_path:
            backbone = vit
            break 
    sam_model = sam_model_registry[backbone](checkpoint=sam_checkpoint_path)

    sam = InContextSAM(
        prompt_encoder=prompt_encoder,
        sam_model=sam_model,
    )
    return sam, clip_prossesor

if __name__ == "__main__":
    sam, clip_prossesor = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        clip_cache_dir=None, 
        sam_checkpoint_path="/root/yunzhi/checkpoint/sam/sam_vit_l_0b3195.pth",
    )
    few_shot_prompt = torch.randn((4, 5, 3, 224, 224)).cuda()    # 4-way 5-shot
    batch_images = torch.randn((3, 3, 1024, 1024)).cuda()
    sam = sam.cuda() 
    sam.eval()
    masks, iou_predictions, low_res_masks = None, None, None  
    with torch.no_grad():
        masks, iou_predictions, low_res_masks = sam(few_shot_prompt, batch_images)
        pdb.set_trace()
        print("dummy")


