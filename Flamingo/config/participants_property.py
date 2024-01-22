""" 
    config
"""
import os.path as osp
from Flamingo.utils import path_finder

# cache_dir = path_finder(CACHE_DIRS)
cache_dir = None 
# lang_encoder_path = osp.join(cache_dir, "models--anas-awadalla--mpt-1b-redpajama-200b/snapshots/50d6bc94e17812873f39c36c5f815263fa71fb80")
lang_encoder_path = "facebook/opt-125m"
tokenizer_path = lang_encoder_path

model_config = dict(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path=lang_encoder_path,
    tokenizer_path=tokenizer_path,
    cross_attn_every_n_layers=1,
    cache_dir = cache_dir,
    lora_tuning=False,
    decoupled=True  
)

#  data_dir, anno_file, tokenizer
dataset_config = dict(
    type='CachedParticipants',
    data_dir='/root/datasets/cached_participants_property/pth',
    anno_file='/root/datasets/cached_participants_property/annotations/train.json',
)
workflows = [('train', 100), ('test', 1)]
# padded_samples = self.tokenizer.pad(x,return_tensors="pt",padding="longest",truncation=True)