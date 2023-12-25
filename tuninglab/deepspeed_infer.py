import deepspeed
import torch
import pdb 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch 
from transformers import AutoModelForSeq2SeqLM


"""
    650 MiB
    550 MiB
"""
def to_cuda(data, device, dtype=torch.float16):
    data['input_ids'] = data['input_ids'].to(device)
    data['attention_mask'] = data['attention_mask'].to(device)
    return data 
"""
    inference test:
"""
model_id="google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
ds_engine = deepspeed.init_inference(model,
                                 mp_size=2,
                                 dtype=torch.float16,
                                #  checkpoint=None if args.pre_load_checkpoint else args.checkpoint_json,
                                 replace_with_kernel_inject=True)
instruction = "tell me some information about China"
seq = to_cuda(data=tokenizer(instruction, return_tensors="pt"), device=model.device)
instruction = tokenizer.decode(model.generate(**seq)[0], skip_special_tokens=False)
# Initialize the DeepSpeed-Inference engine

model = ds_engine.module
output = model('Input String')