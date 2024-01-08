""" 
    inference with DeepSpeed
"""
import pdb 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch 
from transformers import AutoModelForSeq2SeqLM
from rich import print


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
model = model.half()
model = model.cuda()
print(model.encoder.block[0].layer[0].SelfAttention.q.weight)
instruction = ["tell me some information about China", "What is USA? "]
seq = to_cuda(data=tokenizer(instruction, return_tensors="pt", padding="longest"), device=model.device)
print("[@rank{rank}] seq: {seq}, type: {T}".format(rank=-1, seq=str(seq), T=type(seq).__name__))
samples = model.generate(**seq)
# instruction = tokenizer.decode(samples, skip_special_tokens=False)
print("[@rank{rank}] instruction : {instruction}".format(rank=-1, instruction=instruction))
pdb.set_trace()