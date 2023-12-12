""" 
reference :
    1) implementation: Efficient Large Language Model training with LoRA and Hugging Face
        - https://www.philschmid.de/fine-tune-flan-t5-peft

    2) Quant: A Gentle Introduction to 8-bit Matrix Multiplication for transformers at
      scale using Hugging Face Transformers,
      Accelerate and bitsandbytes
        - https://huggingface.co/blog/hf-bitsandbytes-integration

    3) tuning: 
        - https://zhuanlan.zhihu.com/p/637717693

    requirements:
    py7zr
"""
# import evaluate
import pdb 
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import concatenate_datasets
import numpy as np
import torch 
from transformers import AutoModelForSeq2SeqLM
import os 
import os.path as osp 
from Flamingo.utils.pretty import pretty_print, vis_model
import traceback
# Load dataset from the hub
# Train dataset size: 14732
# Test dataset size: 819
model_id="google/flan-t5-small"
save_directory = "/home/yunzhi/yunzhi/yunzhi/checkpoints/flan-t5"

dataset = load_dataset("samsum")
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")



print("model_id: ", model_id)
# Load tokenizer of FLAN-t5-XL
tokenizer = AutoTokenizer.from_pretrained(model_id)
# print("tokenizer: \n", tokenizer)

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"],
                                          dataset["test"]]).map(lambda x: tokenizer(x["dialogue"],
                                            truncation=True), batched=True,
                                              remove_columns=["dialogue", "summary"])
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# take 85 percentile of max length for better utilization
max_source_length = int(np.percentile(input_lenghts, 85))
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"],
                                           dataset["test"]]).map(lambda x: tokenizer(x["summary"],
                                            truncation=True),
                                        batched=True, remove_columns=["dialogue", "summary"])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# take 90 percentile of max length for better utilization
max_target_length = int(np.percentile(target_lenghts, 90))
print(f"Max target length: {max_target_length}")

def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample["dialogue"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# save datasets to disk for later easy loading
tokenized_dataset["train"].save_to_disk("data/train")
tokenized_dataset["test"].save_to_disk("data/eval")

pretty_print(f"start load FP16 model from: {save_directory}")
try: 
    model = AutoModelForSeq2SeqLM.from_pretrained(save_directory,
                                                   load_in_8bit=True,
                                                     device_map="auto")
except OSError:
# except Exception as e:
#     traceback.print_exc()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model = model.half()
    model.save_pretrained(save_directory)
    # torch.save(model.state_dict(), checkpoint)
    print("model saved !")
vis_model(model)
pretty_print("\n -----------------------------------\n 8 bit weight:  \n", color="green")
pretty_print("model.encoder.block[0].layer[0].SelfAttention.q.weight", color="green")
print(model.encoder.block[0].layer[0].SelfAttention.q.weight)
pdb.set_trace()
