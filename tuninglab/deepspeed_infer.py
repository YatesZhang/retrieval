from typing import Any
import deepspeed
import torch
import pdb 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch 
from transformers import AutoModelForSeq2SeqLM
from rich import print
from torch.utils.data import Dataset, DataLoader

"""
    python infer.py

    FP32: 650 MiB
    FP16: 550 MiB
    deepspeed --num_gpus 4 deepspeed_infer.py
    GPU cost:   578MiB per GPU
"""
def to_cuda(data, device, dtype=torch.float16):
    data['input_ids'] = data['input_ids'].to(device)
    data['attention_mask'] = data['attention_mask'].to(device)
    return data 
class InstructionsDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.instructions = [
            "tell me some information about China",
            "What is USA ?"
            "What's the relationship between Google and OpenAI ?",
            "1 + 1 equals ?"
        ]
    def __len__(self):
        return len(self.instructions)
    def __getitem__(self, index):
        return self.instructions[index]

class Collate(object):
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch = self.tokenizer.pad(
            batch,
            return_tensors='pt',
            padding='longest'
        )
        return batch
"""
    inference test:
"""
model_id="google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
dataloader = DataLoader(dataset=InstructionsDataset(), batch_size=1, collate_fn=Collate(tokenizer=tokenizer))
# tokenizer.pad()
ds_engine = deepspeed.init_inference(model,
                                 mp_size=4,
                                 dtype=torch.float16,
                                 data_loader=dataloader,
                                #  checkpoint=None if args.pre_load_checkpoint else args.checkpoint_json,
                                 replace_with_kernel_inject=True 
                                 )
instruction = "tell me some information about China"
seq = to_cuda(data=tokenizer(instruction, return_tensors="pt"), device=model.device)
instruction = tokenizer.decode(model.generate(**seq)[0], skip_special_tokens=False)
# Initialize the DeepSpeed-Inference engine

model = ds_engine.module
# print(model.encoder.block[0].layer[0].SelfAttention.q.weight)
instruction = "tell me some information about China"
seq = to_cuda(data=tokenizer(instruction, return_tensors="pt"), device=model.device)
rank = torch.distributed.get_rank() 
print("[@rank{rank}] seq: {seq}, type: {T}".format(rank=rank, seq=str(seq), T=type(seq).__name__))
instruction = tokenizer.decode(model.generate(**seq)[0], skip_special_tokens=False)
print("[@rank{rank}] instruction : {instruction}".format(rank=rank, instruction=instruction))
# print(instruction)
# pdb.set_trace()
# output = model('Input String')