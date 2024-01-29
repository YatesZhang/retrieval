import math 
import torch 
import deepspeed
# model config and dataset config: 
try:
    import Flamingo
except ModuleNotFoundError:
    import sys
    sys.path.append("..")
    import Flamingo

""" 
    import your config file:
"""
# from Flamingo.config.baseline import dataset_config, model_config, workflows
from Flamingo.config.participants_property import dataset_config_test, model_config_test, workflows
# model:
from Flamingo.lora_tuning import create_model_and_transforms 
from Flamingo.models.batchprocessor import FlamingoBatchProcessor, DecoupledFlamingoBatchProcessor
# DataLoader, DataSampler
from torch.utils.data import DataLoader, DistributedSampler
from Flamingo.datasets import InfiniteSampler, build_dataset
from Flamingo.structure import collate_fn
from Flamingo.inference.post_process import post_process_participants
import pdb 
from tqdm import tqdm 
import os 

os.environ["TRANSFORMERS_OFFLINE"] = "1"
def main():
    """ 
        CUDA_VISIBLE_DEVICES=7 python inference.py 
    """
    model, image_processor, tokenizer = create_model_and_transforms(
        **model_config_test
    )
    """ 
        load state dict:
    """
    load_from = "/root/yunzhi/flamingo_retrieval/retrieval/work_dir/99/weight.pth"
    print("load state dict from: {}".format(load_from))
    state_dict = torch.load(load_from)
    keys1 = model.lang_encoder.gated_cross_attn_layers.load_state_dict(state_dict, strict=False)
    keys2 = model.perceiver.load_state_dict(state_dict, strict=False)

    print("cast to bfloat16 ")
    cast_type = torch.bfloat16
    model = model.to(torch.device("cuda"), dtype=cast_type)
    model.eval()

    dataset = build_dataset(
        dataset_config=dataset_config_test,
        vis_processor=image_processor,
        tokenizer=tokenizer,
    )
    batch_processor = FlamingoBatchProcessor(
         tokenizer=tokenizer, image_processor=image_processor, cast_type=cast_type, num_beams=3, max_new_tokens=20
    )
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=False, num_workers=4)

    acc_cat = 0
    acc_att = 0
    tp_cat = 0
    tp_att = 0
    N = 0
    desc = "acc category: {}, acc attribute: {}".format(acc_cat, acc_att)
    pbar = tqdm(dataloader, desc=desc)
    for batch in pbar:
        """
        dict(
            imgs=imgs,
            category_names=category_names,
            attributes_names=attributes_names,
            metas=metas,
        )
        """ 
        if len(batch['imgs']) == 0:
            continue
        out = batch_processor(model, batch, text_prompt="<image>Output:", mode='test', num_beams=3, max_new_tokens=20)
        # print("before post process:", out)
        out = post_process_participants(out)
        # print("after post process:", out)
        category_names=batch['category_names']
        attributes_names=batch['attributes_names']
        metas = batch['metas']
        # print("category_names:", category_names)
        # print("attributes_names:", attributes_names)
        # pdb.set_trace()
        for j in range(len(out)):
            N += 1
            pred = out[j]
            category_name = category_names[j]
            attributes_name = attributes_names[j]
            flag = False
            if category_name in pred:
                tp_cat += 1
            else:
                flag = True
            if attributes_name in pred:
                tp_att += 1
            else:
                flag = True
            if flag:
                print([attributes_name, category_name], 'is predict to ', pred)
        acc_cat = tp_cat / N
        acc_att = tp_att / N
        desc = "acc category: {}, acc attribute: {}".format(acc_cat, acc_att)
        pbar.set_description(desc)
        print(desc)
    # pdb.set_trace()

if __name__ == "__main__":
    main()