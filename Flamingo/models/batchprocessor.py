""" 
    flamingo 
"""
from typing import Any
import torch 
from rich import print
import numpy as np 
import PIL
from PIL import Image
import pdb 
from copy import deepcopy
PRECISIONS = {
    "fp32": torch.float,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
def img_auto_cast(imgs):
    """ 
        clip vision_encoder's image processor accept Pillow image only
    """

    if isinstance(imgs, np.ndarray):
        return Image.fromarray(imgs)
    if isinstance(imgs, Image.Image):
        return imgs 
    if isinstance(imgs, list):
        return [img_auto_cast(img) for img in imgs]
    if isinstance(imgs, torch.Tensor):
        """ 
            directly return a torch Tensor
        """
        return imgs 
    if isinstance(imgs, dict):
        possible_key_list = [
            'img', 'image', 'imgs', 'images'
        ]
        for img_key in possible_key_list:
            has_key = img_key in imgs 
            if has_key:
                return imgs[img_key]
            else:
                continue 
        raise KeyError("{} should in dict input!".format(str(possible_key_list)))

# from torch.cuda.amp import autocast
class FlamingoBatchProcessor(object):
    def __init__(self, tokenizer=None, image_processor=None, cast_type=torch.bfloat16, num_beams=3, max_new_tokens=20):
        """ 
            training with gradient accumulation
            (integrated in DeepSpeed)

            usage: 
                train: 
        """
        self.tokenizer = tokenizer 
        self.cast_type = cast_type
        self.num_beams = num_beams 
        self.max_new_tokens = max_new_tokens
        self.image_processor = image_processor 
        # self.gradient_accumulation_steps = gradient_accumulation_steps
        pass 

    def get_device(self, model):
        """
            get model's device
        """
        if hasattr(model, 'device'):
            return model.device 
        device = next(model.parameters()).device
        return device  
    
    def process_batch(self, model, batch, few_shot_prompt=None, **kwargs):
        """ 
            training phase
        """
        device = self.get_device(model=model) 
        images = batch["image"].to(device, dtype=self.cast_type, non_blocking=True).unsqueeze(1).unsqueeze(1)
        input_ids = batch["input_ids"].to(device, dtype=torch.long,  non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.long,  non_blocking=True)
        labels = batch["labels"].to(device, dtype=torch.long, non_blocking=True)
        
        loss = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )[0]
        
        # sum loss: 
        loss = loss.sum() 
        # loss /= self.gradient_accumulation_steps 
        return loss
    
    def image_auto_cast(self, imgs):
        """ 
            return imgs of shape [B, C, H, W]
        """
        if isinstance(imgs, torch.Tensor):
            while len(imgs.shape) < 4:
                imgs = imgs.unsqueeze(0)
            return imgs 
        elif isinstance(imgs, PIL.Image.Image):
            return self.image_processor(imgs).unsqueeze(0)    # shape: [1, C, H, W]
        elif isinstance(imgs, list):
            return torch.cat([self.image_auto_cast(img) for img in imgs], dim=0)
        else:
            raise TypeError

    @torch.inference_mode()
    def inference(self, model, batch, text_prompt="", num_beams=-1, max_new_tokens=-1, **kwargs):
        device = self.get_device(model=model)
        img = None
        if isinstance(batch, dict):
            for key in ['img', 'image', 'imgs', 'images']:
                if key in batch:
                    img = batch[key]
                    break
        else:
            img = batch
        img = self.image_auto_cast(img)    # shape: [B, C, H, W]
        img = img.unsqueeze(1).unsqueeze(1)   # shape: [B, T=1, F=1, C, H, W]
        
        assert isinstance(img, torch.Tensor) and len(img.shape) == 6, "img should be a 6-dim tensor as [B, T, F. C, H, W]"
        img = img.to(device, dtype=self.cast_type, non_blocking=True)
        if text_prompt == "":
            text_prompt = "<image>Output:"
        elif not text_prompt.startswith("<image>"):
            text_prompt = "<image>{}".format(text_prompt)
        batch_size = img.shape[0]
        batch_encoding = self.tokenizer(text_prompt, return_tensors="pt", padding=True)
        input_ids = batch_encoding["input_ids"].to(device, dtype=torch.long, non_blocking=True).repeat(batch_size, 1)
        attention_mask = batch_encoding["attention_mask"].to(device, dtype=torch.long, non_blocking=True).repeat(batch_size, 1)
        generated_text = model.generate(
            vision_x=img,
            lang_x=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens if max_new_tokens > 0 else self.max_new_tokens,
            num_beams=num_beams if num_beams > 0 else self.num_beams,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_text = generated_text.cpu()
        return self.tokenizer.batch_decode(generated_text, skip_special_tokens=True)


    def __call__(self, model, batch, mode='test', **kwargs):
        """ 
            call the batch processor in training  
        """
        assert mode in ['train', 'test', 'inference']
        if mode == 'train':
            return self.process_batch(model, batch, **kwargs)
        elif mode == 'test':
            return self.inference(model, batch, **kwargs)
        else:   # inference:
            raise NotImplementedError


class DecoupledFlamingoBatchProcessor(FlamingoBatchProcessor):
    def __init__(self, tokenizer=None, image_processor=None, cast_type=torch.bfloat16, num_beams=3, max_new_tokens=20):
        """ 
        
        """
        if isinstance(cast_type, str):
            assert cast_type in PRECISIONS
            cast_type = PRECISIONS[cast_type]
        super().__init__(tokenizer=tokenizer,
         image_processor=image_processor, cast_type=cast_type, num_beams=num_beams, max_new_tokens=max_new_tokens)

    def process_batch(self, model, batch, few_shot_prompt=None, **kwargs):
        """ 
            training phase
        """
        device = self.get_device(model=model)
        
        """
            vision_x : [B, T, F, V, D]
        """
        vision_x = batch["vision_x"].to(device, dtype=self.cast_type, non_blocking=True)
        input_ids = batch["input_ids"].to(device, dtype=torch.long,  non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.long,  non_blocking=True)
        labels = batch["labels"].to(device, dtype=torch.long, non_blocking=True)
        
        loss = model(
                vision_x=vision_x,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )[0]
        
        # sum loss: 
        loss = loss.sum() 
        # loss /= self.gradient_accumulation_steps 
        return loss
    
    @torch.inference_mode()
    def inference(self, model, batch, text_prompt="", num_beams=-1, max_new_tokens=-1, **kwargs):
        device = self.get_device(model=model)
        if isinstance(batch, dict):
            img = batch["img"] 
        else:
            img = batch

        assert isinstance(img, torch.Tensor) and len(img.shape) == 5, "img should be a 5-dim tensor as [B, T, F. V, D]"
        img = img.to(device, dtype=self.cast_type, non_blocking=True)
        if text_prompt == "":
            text_prompt = "<image>Output:"
        elif not text_prompt.startswith("<image>"):
            text_prompt = "<image>{}".format(text_prompt)
        batch_size = img.shape[0]
        batch_encoding = self.tokenizer(text_prompt, return_tensors="pt", padding=True)
        input_ids = batch_encoding["input_ids"].to(device, dtype=torch.long, non_blocking=True).repeat(batch_size, 1)
        attention_mask = batch_encoding["attention_mask"].to(device, dtype=torch.long, non_blocking=True).repeat(batch_size, 1)
        generated_text = model.generate(
            vision_x=img,
            lang_x=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens if max_new_tokens > 0 else self.max_new_tokens,
            num_beams=num_beams if num_beams > 0 else self.num_beams,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_text = generated_text.cpu()
        return self.tokenizer.batch_decode(generated_text, skip_special_tokens=True)
       
class CLIPBatchProcessor(object):
    def __init__(self, vision_encoder, image_processor):
        """
        usage:
            clip_vision_encoder_path="ViT-L-14"
            clip_vision_encoder_pretrained="openai"
            cross_attn_every_n_layers=1
            cache_dir = "/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo"
            vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
                clip_vision_encoder_path,    # "ViT-L-14"
                pretrained=clip_vision_encoder_pretrained,    # "openai"
                cache_dir=cache_dir,
            )

            batch_processor = CLIPBatchProcessor(vision_encoder, image_processor)
        """
        # set the vision encoder to output the visual features
        
        if hasattr(vision_encoder, 'visual'):
            vision_encoder.visual.output_tokens = True
            self.vision_encoder = vision_encoder.visual 
        else:
            vision_encoder.output_tokens = True 
            self.vision_encoder = vision_encoder
        self.image_processor = image_processor
        
        # evaluate mode
        self.vision_encoder.eval()

    def to(self, device):
        """ 
            change the device of vision_encoder to device
        """
        self.vision_encoder = self.vision_encoder.to(device)
        self.device = device
        return self

    def get_device(self, model):
        """
            get model's device
        """
        if hasattr(model, 'device'):
            return model.device 
        device = next(model.parameters()).device
        return device  
    
    def img_auto_collect(self, imgs):
        """ 
            accept input from img_auto_cast
            auto ignore Tensor input when calling image_processor
            input type: 
                list: [PIL.Image]
                PIL.Image
            output:
                Tensor of shape: [B, C, H, W]
        """
        if isinstance(imgs, Image.Image):
            """ 
                PIL.Image -> Tensor [C, H, W] -> [1, C, H, W]
            """
            imgs = self.image_processor(imgs)
            return imgs[None, ...]

        if isinstance(imgs, torch.Tensor):
            """ 
                ignore imgs input
            """
            if len(imgs.shape) == 3:
                # [C, H, W]
                imgs = imgs[None, ...]
            assert imgs.shape == 4, "Tensor of imgs should be [C, H, W] or [B, C, H, W]!"
            return imgs 

        if isinstance(imgs, list):
            """
                [PIL.Image] -> Tensor [B, C, H, W]
            """
            imgs = [self.image_processor(img) for img in imgs]
            if len(imgs[0].shape) == 3:
                imgs = [img[None, ...] for img in imgs]
            elif len(imgs[0].shape) == 4:
                pass 
            else:
                raise ValueError
            return torch.cat(imgs, dim=0)
        
        raise TypeError("input type should be PIL, not {}".format(type(imgs).__name__))
    
    @torch.inference_mode()
    def try_batch(self, imgs):
        """
            try batch process the input images
            input:
                imgs: [B, C, H, W]
            output:
                out: [B, 1, 1, v, d]
        """
        if imgs.shape[0] == 1:
            try:
                out = self.vision_encoder(imgs)[1]
            except RuntimeError:
                print("[warning!] try cpu offload")
                if not hasattr(self, 'vision_encoder_cpu'):
                    self.vision_encoder_cpu = deepcopy(self.vision_encoder).cpu().eval()
                out = self.vision_encoder_cpu(imgs.cpu())[1]    # shape: [B, V, D]
                out = out.unsqueeze(1).unsqueeze(1) 
                return out
        else:
            try:
                mid = imgs.shape[0] // 2
                out_l = self.vision_encoder(imgs[:mid])[1] # shape: [B//2, V, D]
                out_r = self.vision_encoder(imgs[mid:])[1] # shape: [B//2, V, D]
            except RuntimeError:
                out_l = self.try_batch(imgs[:mid])
                out_r = self.try_batch(imgs[mid:])
            out = torch.cat([out_l, out_r], dim=0)
            while len(out.shape) < 5:
                out = out.unsqueeze(1)
        return out

    @torch.inference_mode()
    def __call__(self, imgs):
        """
        support: 
        single: 
            read Pillow Image
            numpy image
        batch:
            from list of Pillow Image or ndarray
            from [B, C, H, W]
            
        """

        # get imgs:
        imgs = img_auto_cast(imgs)
        imgs = self.img_auto_collect(imgs)
        assert isinstance(imgs, torch.Tensor) and len(imgs.shape) == 4 
        
        # to device:
        device = self.get_device(model=self.vision_encoder)
        imgs = imgs.to(device)
        """ 
        vision_encoder:
            vision_encoder(img)[0].shape: torch.Size([1, 768])
            vision_encoder(img)[1].shape: torch.Size([1, 256, 1024])
        """
        out = None
        try:
            out = self.vision_encoder(imgs)[1]
            out = out.unsqueeze(1).unsqueeze(1)
        except RuntimeError:
            print("[warning!] try cpu offload")
            out = self.try_batch(imgs)
        # out shape: (B, T, F) v, d == [B, 1, 1, v, d]
        return out 