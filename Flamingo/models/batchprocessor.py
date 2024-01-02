""" 
    flamingo 
"""
from typing import Any
import torch 
from rich import print
import pdb 


# from torch.cuda.amp import autocast
class FlamingoBatchProcessor(object):
    def __init__(self, tokenizer=None, cast_type=torch.bfloat16, num_beams=3, max_new_tokens=20):
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
    
    def process_batch(self, model, batch, few_shot_prompt=None):
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
     
    # @torch.no_grad
    def inference(self, model, batch):
        """ 
            batch_size x num_media x num_frames x channels x height x width. 
            B, N, F, C, H, W

            batch['input_ids'] : [2, 151]
            batch['labels']    : [2, 151]
        """
        device = self.get_device(model=model) 
        images = batch["image"].to(device, dtype=self.cast_type, non_blocking=True)
        if len(images.shape) == 4:
            images = images.unsqueeze(1).unsqueeze(1)
        
        input_ids = batch['input_ids'].to(device, dtype=torch.long, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.long,  non_blocking=True)
        # labels = batch["labels"].to(device, dtype=torch.long, non_blocking=True)

        generated_text = model.generate(
            vision_x=images,
            lang_x=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_text = generated_text.cpu()
        result_text = [self.tokenizer.decode(text, skip_special_tokens=True) for text in generated_text]
        return result_text


    def __call__(self, model, batch, mode='train'):
        """ 
            call the batch processor in training  
        """
        assert mode in ['train', 'test', 'inference']
        if mode == 'train':
            return self.process_batch(model, batch)
        elif mode == 'test':
            return self.inference(model, batch)
        else:   # inference:
            raise NotImplementedError