import open_clip
from PIL import Image 
import numpy as np 
import pdb 
import torch 
from Flamingo.lora_tuning import create_model_and_transforms
from Flamingo.config.baseline import model_config 
from transformers import BatchEncoding 

# from open_flamingo import create_model_and_transforms
def to_cuda(data, device=0):
    if isinstance(data, BatchEncoding):
        # k: ['input_ids', 'attention_mask']
        for k in data:
            data[k] = data[k].cuda(device)
    return data 

if __name__ == "__main__":
    clip_vision_encoder_path="ViT-L-14"
    clip_vision_encoder_pretrained="openai"
    cross_attn_every_n_layers=1
    cache_dir = "/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo"
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path,    # "ViT-L-14"
        pretrained=clip_vision_encoder_pretrained,    # "openai"
        cache_dir=cache_dir,
    )
    # set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True
    
    img = Image.open("/home/yunzhi/yunzhi/datasets/meta/1.png")
    # input: Pillow format image
    pdb.set_trace()
    img = image_processor(img)
    img = img[None, ...].cuda()    # [1, C, H, W]
    encoder = vision_encoder.visual 
    encoder = encoder.cuda()
    encoder.eval()
    """ 
    encoder:
        encoder(img)[0].shape: torch.Size([1, 768])
        encoder(img)[1].shape: torch.Size([1, 256, 1024])
    """
    with torch.no_grad():
        out = encoder(img)[1]
        out = out.unsqueeze(1).unsqueeze(1)
        print("(B, T, F) v, d | output[1] from CLIP.visual_encoder.vision", out.shape)
    
    model_config['decoupled'] = True
    model_config['lora_tuning'] = False
    model, image_processor, tokenizer = create_model_and_transforms(
        **model_config
    )
    model = model.cuda()
    lang_x = tokenizer(
        ["<image>The color is"],
        return_tensors="pt",
    )
    lang_x = to_cuda(lang_x, device=0)
    with torch.inference_mode():
        text = model.generate(
                        vision_x=out ,
                       lang_x=lang_x["input_ids"],
                        attention_mask=lang_x["attention_mask"],
                        max_new_tokens=20,
                        num_beams=3)
        loss = model(vision_x=out, lang_x=lang_x["input_ids"], attention_mask=lang_x["attention_mask"], labels=lang_x["input_ids"])
        print(tokenizer.batch_decode(text))
    pdb.set_trace()
    