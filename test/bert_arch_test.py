try:
    import Flamingo
except ModuleNotFoundError:
    import sys
    sys.path.append("..")
    import Flamingo
from Flamingo.lora_tuning import create_model_and_transforms 
from PIL import Image
from rich import print 
import pdb 
import torch 

if __name__ == "__main__":
    """ 
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python bert_arch_test.py
        get model: 
    """
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="bert-base-uncased",
        tokenizer_path="bert-base-uncased",
        cross_attn_every_n_layers=1,
        use_local_files=True,
        # decoder_layers_attr_name=None,
        # freeze_lm_embeddings=False,
        # cache_dir=None,
        # lora_tuning=False,
        # add_eos_token=True,
        decoupled=False)
    model.eval()
    """ 
        input imgs:
    """
    img = Image.open("/root/yunzhi/flamingo_retrieval/retrieval/Flamingo/images/yellow_bus.jpg")
    lang_x = ["<image>Output:[MASK]"]
    lang_x = tokenizer(lang_x, return_tensors="pt", padding=True)
    vision_x = image_processor(img).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = model(
            vision_x=vision_x,
            lang_x=lang_x['input_ids'], 
            attention_mask=lang_x['attention_mask'],
            labels=lang_x['input_ids'].detach().clone())
        print(out)
    """
        classifier token and separattor token should not be masked
    """
    print("special tokens:[CLS]=101, [SEP]=102, [MASK]=103")
    print(tokenizer.decode([101, 102, 103]))
    pdb.set_trace()