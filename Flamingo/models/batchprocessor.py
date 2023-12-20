from typing import Any
import torch 
# from torch.cuda.amp import autocast
class FlamingoBatchProcessor(object):
    def __init__(self, cast_type=torch.bfloat16):
        """ 
            training with gradient accumulation
            (integrated in DeepSpeed)
        """
        self.cast_type = cast_type
        # self.gradient_accumulation_steps = gradient_accumulation_steps
        pass 

    def process_batch(self, model, batch):
        """ 
            training phase
        """
        # images = batch["image"].to(device_id, dtype=cast_dtype, non_blocking=True)\
        # .unsqueeze(1).unsqueeze(1)
        # input_ids = batch["input_ids"].to(device_id,
        #                                    dtype=cast_dtype, non_blocking=True)
        # attention_mask = batch["attention_mask"].to(device_id,
        #                                              dtype=cast_dtype,
        #                                                non_blocking=True)
        # labels = batch["labels"].to(device_id, dtype=cast_dtype, non_blocking=True)
        device = model.device 
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

    def __call__(self, model, batch, mode='train'):
        """ 
            call the batch processor in training  
        """
        assert mode in ['train', 'test', 'inference']
        if mode == 'train':
            return self.process_batch(model, batch)
        elif model == 'test':
            raise NotImplementedError
        else:   # inference:
            raise NotImplementedError