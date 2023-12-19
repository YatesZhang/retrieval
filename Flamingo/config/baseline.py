import os.path as osp
# .13:
# cache_dir = "/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo"
# lang_encoder_path = "anas-awadalla/mpt-1b-redpajama-200b"

# .89:
cache_dir = "/root/ln_homework/code/third_party/VLLM/retrieval/Flamingo/cache_dir/flamingo"
lang_encoder_path = osp.join(cache_dir, "models--anas-awadalla--mpt-1b-redpajama-200b")
tokenizer_path = lang_encoder_path

model_config = dict(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path=lang_encoder_path,
    tokenizer_path=tokenizer_path,
    cross_attn_every_n_layers=1,
    cache_dir = cache_dir,
    lora_tuning=True  
)

# .13
# vis_root = "/home/yunzhi/datasets/COCO/train2017"
# .89:
vis_root = "/root/ln_homework/data/COCO/train2017"
anno_path = "/root/ln_homework/data/COCO/aokvqa_v1p0/aokvqa_v1p0_train.json"
dataset_config = dict(
    type="aokvqa",
    vis_root=vis_root,
    ann_paths=[anno_path],
    sample_image=False,
)

workflows = [('train', 20), ('test', 1)]
# padded_samples = self.tokenizer.pad(x,return_tensors="pt",padding="longest",truncation=True)