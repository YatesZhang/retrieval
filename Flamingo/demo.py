"""
    quick demo for openflamingo
"""

# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch
# from open_flamingo import create_model_and_transforms
from Flamingo.lora_tuning import create_model_and_transforms
from PIL import Image
import requests
import torch
from transformers.tokenization_utils_base import BatchEncoding
import os.path as osp
from Flamingo.utils import path_finder

CACHE_DIRS = [
    # .13:
    "/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo",
    # .89:
    "/root/ln_homework/code/third_party/VLLM/retrieval/Flamingo/cache_dir/flamingo",
    # .7
    "/cfsdata2/cfsdata/workspace/yunzhi/checkpoint/flamingo"
]
cache_dir = path_finder(CACHE_DIRS)
# for dir_path in CACHE_DIRS:
#     if osp.exists(cache_dir):
# cache_dir = "/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo"
# lang_encoder_path = "anas-awadalla/mpt-1b-redpajama-200b"
# .89:
# cache_dir = "/root/ln_homework/code/third_party/VLLM/retrieval/Flamingo/cache_dir/flamingo"
lang_encoder_path = "facebook/opt-125m"
# lang_encoder_path = "bert-base-uncased"
tokenizer_path = lang_encoder_path
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path=lang_encoder_path,
    tokenizer_path=tokenizer_path,
    cross_attn_every_n_layers=1,
    cache_dir= cache_dir,  # Defaults to ~/.cache
    lora_tuning=False
)

"""
Step 1: Load images
"""
# demo_image_one = Image.open(
#     requests.get(
#         "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
#     ).raw
# )

# demo_image_two = Image.open(
#     requests.get(
#         "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
#         stream=True
#     ).raw
# )

# query_image = Image.open(
#     requests.get(
#         "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
#         stream=True
#     ).raw
# )
demo_image_two = Image.open("./images/2.jpg")
demo_image_one = Image.open("./images/1.jpg")
# demo_image_one = Image.open("./images/night.jpg")
# query_image = Image.open("./images/snow.jpg")
# query_image = Image.open("./images/night.jpg")
query_image = Image.open("./images/yellow_bus.jpg")
def to_cuda(data, device=0):
    if isinstance(data, BatchEncoding):
        # k: ['input_ids', 'attention_mask']
        for k in data:
            data[k] = data[k].cuda(device)
    return data 

if __name__ == "__main__":
    device = 1
    model = model.cuda(device)

    """
    Step 2: Preprocessing images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
    batch_size x num_media x num_frames x channels x height x width. 
    In this case batch_size = 1, num_media = 3, num_frames = 1,
    channels = 3, height = 224, width = 224.
    """
    vision_x = [image_processor(demo_image_one).unsqueeze(0),
        image_processor(demo_image_two).unsqueeze(0),
        image_processor(query_image).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0).cuda(device)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    print(vision_x.shape)

    """
    Step 3: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
    We also expect an <|endofchunk|> special token to indicate the end of the text 
    portion associated with an image.
    """
    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    lang_x = tokenizer(
        ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom.<|endofchunk|><image>the color is"],
        return_tensors="pt",
    )
    lang_x = to_cuda(lang_x, device=device)

    """
    Step 4: Generate text
    """
    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=20,
        num_beams=3,
    )
    print("Generated text: ", tokenizer.decode(generated_text[0]))