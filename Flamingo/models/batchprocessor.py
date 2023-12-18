from typing import Any


class FlamingoBatchProcessor(object):
    def __init__(self, gradient_accumulation_steps=1.0):
        """ 
            training with gradient accumulation
            (integrated in DeepSpeed)
        """
        # self.gradient_accumulation_steps = gradient_accumulation_steps
        pass 

    def process_batch(self, model, batch):
        # images = batch["image"].to(device_id, dtype=cast_dtype, non_blocking=True)\
        # .unsqueeze(1).unsqueeze(1)
        # input_ids = batch["input_ids"].to(device_id,
        #                                    dtype=cast_dtype, non_blocking=True)
        # attention_mask = batch["attention_mask"].to(device_id,
        #                                              dtype=cast_dtype,
        #                                                non_blocking=True)
        # labels = batch["labels"].to(device_id, dtype=cast_dtype, non_blocking=True)
        device = model.device 
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # in DeepSpeed: 
        # auto deployment on GPU
        # auto cast to fp16
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