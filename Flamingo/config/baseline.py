model_config = dict(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
    cache_dir = "/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo",
    lora_tuning=True  
)

dataset_config = dict(
    type="aokvqa",
    vis_root="/home/yunzhi/datasets/COCO/train2017",
    ann_paths=["/home/yunzhi/datasets/aokvqa_v1p0/aokvqa_v1p0_train.json"],
    sample_image=False,
)


# padded_samples = self.tokenizer.pad(x,return_tensors="pt",padding="longest",truncation=True)