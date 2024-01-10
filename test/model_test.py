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
from Flamingo.lora_tuning import create_model_and_transforms 
import pdb
import deepspeed
if __name__ == "__main__":
    """ 
    Debug model with deepspeed. 
    Example: 
        deepspeed --num_gpus=1 --local_rank=0 model_test.py 
    
    """
    # build model, image processor and tokenizer
    model, image_processor, tokenizer = create_model_and_transforms(
        **model_config
    )
    # model = deepspeed.initialize(model=model)
    pdb.set_trace()