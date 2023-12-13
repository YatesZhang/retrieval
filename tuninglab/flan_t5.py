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
    
    run command: 
        CUDA_VISIBLE_DEVICES=0 python flan_t5.py
    ---------------------------------------------
    enable_int8 = True
    GPU Memory: 876MiB
    time cost: 10 min/epoch
    ---------------------------------------------
    enable_int8 = False
    train: epoch = 1
    GPU Memory: 2104MiB
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
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Load dataset from the hub
# Train dataset size: 14732
# Test dataset size: 819
model_id="google/flan-t5-small"
save_directory = "/home/yunzhi/yunzhi/yunzhi/checkpoints/flan-t5"
work_dir = "/home/yunzhi/yunzhi/yunzhi/VLLM/retrieval/work_dir"
enable_int8 = True

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


# Load model as int8:

try:
    if enable_int8: 
        pretty_print(f"start load FP16 model from: {save_directory} to int8")
        model = AutoModelForSeq2SeqLM.from_pretrained(save_directory,
                                                    load_in_8bit=enable_int8,
                                                        device_map="auto")
    else:
        pretty_print(f"start load FP32 model from: {model_id}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    # if not enable_int8:
    #     model = model.half
except OSError:
# except Exception as e:
#     traceback.print_exc()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model = model.half()
    model.save_pretrained(save_directory)
    # torch.save(model.state_dict(), checkpoint)
    print("model saved !")

# Define LoRA Config
lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)
# prepare int-8 model for training
if enable_int8:
    model = prepare_model_for_int8_training(model)
# else:
#     model = model.half()   please use: pytorch_lightning Automatic Mixed Precision，AMP
# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

weight_q = model.encoder.block[0].layer[0].SelfAttention.q.weight
pretty_print("\n -----------------------------------\n {} pretraining weight:  \n".format(weight_q.dtype),
              color="green")
pretty_print("model.encoder.block[0].layer[0].SelfAttention.q.weight", color="green")
print(weight_q)

weight_lora = model.encoder.block[0].layer[0].SelfAttention.q.lora_A['default'].weight
pretty_print("\n -----------------------------------\n {} LoRA weight:  \n".format(weight_lora.dtype),
              color="green")
pretty_print("model.encoder.block[0].layer[0].SelfAttention.q.weight", color="green")
print(weight_lora)
vis_model(model)
# pdb.set_trace()

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=work_dir,
	auto_find_batch_size=True,
    learning_rate=1e-3, # higher learning rate
    num_train_epochs=1,
    logging_dir="{}/logs".format(work_dir),
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="no",
    # report_to="tensorboard",
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
""" 
Let's now train our model and run the cells below.
Note that for T5, some layers are kept in float32 for stability purposes
"""
# train model
trainer.train()

# Save our LoRA model & tokenizer results
if enable_int8:
    peft_model_id="int8flan-t5-small-fp16LoRA"
else:
    peft_model_id="fp32flan-t5-small-fp32LoRA"
peft_model_id = osp.join(work_dir, peft_model_id)
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
pretty_print("Save pretrained model: {}".format(peft_model_id))
# if you want to save the base model to call
# trainer.model.base_model.save_pretrained(peft_model_id)


