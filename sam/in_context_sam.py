"""
    segment anything in-context
"""

from segment_anything import sam_model_registry
import torch 
from torch import nn 
import open_clip 


class InContextSAM(nn.Module):
    def __init__(
        self, 
        prompt_encoder: nn.Module, 
        sam_model: nn.Module,):
        super().__init__()
        """ 
            embed_dim: 
            image_embedding_size: 
        """
        self.embed_dim = prompt_encoder.embed_dim
        self.image_embedding_size = sam_model.prompt_encoder.image_embedding_size
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
    
    def forward(few_shot_prompt, batch_images):
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
                K * N, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )    # [K * N, 256, 64, 64]

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
            low_res_masks.append(low_res_mask)
            iou_predictions.append(iou_prediction)
        low_res_masks = torch.cat(low_res_masks, dim=0) 
        iou_predictions = torch.cat(iou_predictions, dim=0)
        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        if not return_logits:
            masks = masks > self.mask_threshold

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
):
    prompt_encoder, clip_prossesor = create_prompt_encoder(
        clip_vision_encoder_path=clip_vision_encoder_path,
        clip_vision_encoder_pretrained=clip_vision_encoder_pretrained,
        cache_dir=clip_cache_dir
    )

