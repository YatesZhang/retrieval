# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch
# from open_flamingo import create_model_and_transforms
from lora_tuning import create_model_and_transforms
from PIL import Image
import requests
import torch
from transformers.tokenization_utils_base import BatchEncoding

cache_dir = "/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo"
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
    cache_dir= cache_dir # Defaults to ~/.cache
)

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt", cache_dir=cache_dir)
# checkpoint_path = "/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo/checkpoint.pt"
model.load_state_dict(torch.load(checkpoint_path), strict=False)