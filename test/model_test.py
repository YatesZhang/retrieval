lang_encoder_path = "bert-base-cased"
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
cache_dir = None 
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
import sys 
sys.path.append("..")
import torch
from Flamingo.lora_tuning import create_model_and_transforms 
from Flamingo.models.batchprocessor import DecoupledFlamingoBatchProcessor
from Flamingo.config.baseline import dataset_config
from Flamingo.datasets import build_dataset
import pdb
import deepspeed
if __name__ == "__main__":
    """ 
    Debug model with deepspeed. 
    Example: 
        deepspeed --num_gpus=1 --local_rank=0 model_test.py 
        CUDA_VISIBLE_DEVICES=2 python model_test.py 
    """
    # build model, image processor and tokenizer
    model, image_processor, tokenizer = create_model_and_transforms(
        **model_config
    )
    print("Load state dict:")
    state_dict = torch.load("/root/yunzhi/flamingo_retrieval/retrieval/work_dir/100/weight.pth")
    # pdb.set_trace()
    keys1 = model.lang_encoder.gated_cross_attn_layers.load_state_dict(state_dict, strict=False)
    keys2 = model.perceiver.load_state_dict(state_dict, strict=False)
    dataset = build_dataset(
        dataset_config=dataset_config,
        vis_processor=image_processor,
        tokenizer=tokenizer)
    model.eval()
    batch_processor = DecoupledFlamingoBatchProcessor(cast_type='bf16', tokenizer=tokenizer)
    with torch.no_grad():
        for data in dataset:
            img = data['img']
            path = data['path']
            label = data['label']
            output = batch_processor(model=model, batch=img, mode='test',
            text_prompt="<image>Output:", num_beams=3, max_new_tokens=20)
            pdb.set_trace()