from Flamingo.datasets.builder import build_dataset 
from Flamingo.config.baseline import dataset_config, model_config
from Flamingo.utils.pretty import pretty_print 
from Flamingo.lora_tuning import create_model_and_transforms
import pdb 
# from transformers.models
model_config = dict(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
    cache_dir = "/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo",
    lora_tuning=True,
    add_eos_token=False
)

dataset_config = dict(
    type="aokvqa",
    vis_root="/home/yunzhi/datasets/COCO/train2017",
    ann_paths=["/home/yunzhi/datasets/aokvqa_v1p0/aokvqa_v1p0_train.json"],
    sample_image=False,
)
if __name__ == "__main__":

    _, image_processor, tokenizer = create_model_and_transforms(
        **model_config
    )
    # pdb.set_trace()
    dataset = build_dataset(
        dataset_config=dataset_config,
        vis_processor=image_processor,
        tokenizer=tokenizer,
    )

    data = dataset[0]
    pdb.set_trace()