""" 
    config
"""
import os.path as osp
from Flamingo.utils import path_finder

CACHE_DIRS = [
    # .13:
    "/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo",
    # .89:
    "/root/ln_homework/code/third_party/VLLM/retrieval/Flamingo/cache_dir/flamingo"
]

VIS_ROOTS = [
    "/root/ln_homework/data/COCO/train2017",
    "/home/datasets/COCO/train2017"
]

ANNO_PATHS = [
    "/home/datasets/COCO/annotations/aokvqa_v1p0_train.json",
    "/home/yunzhi/datasets/aokvqa_v1p0/aokvqa_v1p0_train.json"
]
cache_dir = path_finder(CACHE_DIRS)
# lang_encoder_path = osp.join(cache_dir, "models--anas-awadalla--mpt-1b-redpajama-200b/snapshots/50d6bc94e17812873f39c36c5f815263fa71fb80")
lang_encoder_path = "facebook/opt-125m"
tokenizer_path = lang_encoder_path

# model_config = dict(
#     clip_vision_encoder_path="ViT-L-14",
#     clip_vision_encoder_pretrained="openai",
#     lang_encoder_path=lang_encoder_path,
#     tokenizer_path=tokenizer_path,
#     cross_attn_every_n_layers=1,
#     cache_dir = cache_dir,
#     lora_tuning=True  
# )
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
#
vis_root = path_finder(VIS_ROOTS)
anno_path = path_finder(ANNO_PATHS)

# dataset_config = dict(
#     type="aokvqa",
#     vis_root=vis_root,
#     ann_paths=[anno_path],
#     sample_image=False,
# )
dataset_config = dict(
    type='gtsrb',
    data_dir='/home/yunzhi/yunzhi/datasets',
    train_path='/home/yunzhi/yunzhi/datasets/Train',
    test_path='/home/yunzhi/yunzhi/datasets',
    mode='train'
)
workflows = [('train', 100), ('test', 1)]
# padded_samples = self.tokenizer.pad(x,return_tensors="pt",padding="longest",truncation=True)