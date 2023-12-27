"""
    dataset test
    test: 
        A-OK VQA
"""

from Flamingo.datasets.builder import build_dataset 
from Flamingo.config.baseline import dataset_config, model_config
from Flamingo.utils.pretty import pretty_print 
from Flamingo.lora_tuning import create_model_and_transforms
from Flamingo.models import FlamingoBatchProcessor
from torch.utils.data import DataLoader 
import torch 
import pdb 
import deepspeed
# torch.cuda.current_device()
# from transformers.models
model_config = dict(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
    cache_dir = "/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo",
    lora_tuning=True,
    add_eos_token=True 
)

dataset_config = dict(
    type="aokvqa",
    vis_root="/home/yunzhi/datasets/COCO/train2017",
    ann_paths=["/home/yunzhi/datasets/aokvqa_v1p0/aokvqa_v1p0_train.json"],
    sample_image=False,
)


if __name__ == "__main__":
    model, image_processor, tokenizer = create_model_and_transforms(
        **model_config
    )
    # pdb.set_trace()
    state_dict = torch.load("/home/yunzhi/yunzhi/yunzhi/VLLM/retrieval/work_dir/50/loRA.pth")
    keys = model.load_state_dict(state_dict, strict=False)
    # print("Load State Dict: \n", keys)
    model = model.cuda()
    model.eval() 

    dataset = build_dataset(
        dataset_config=dataset_config,
        vis_processor=image_processor,
        tokenizer=tokenizer,
    )
    """ 
        NotImplementedError: MosaicGPT does not support generation with right padding
    """
    tokenizer.padding_side = 'left'
    """ 
        test dataset:
    """
    # for data in dataset:
    #     labels = data['labels']
    #     instruction = data['instruction']
    #     input_ids = data['input_ids']
    #     attentions_mask = data['attention_mask']
    #     image = data['image']
    #     answer = data['answer']
    #     pdb.set_trace()
    """ 
    
    {'split': 'train', 'image_id': 299207, 'question_id': '22MexNkBPpdZGX6sxbxVBH',
    'question': 'What is the man by the bags awaiting?',
    'choices': ['skateboarder', 'train', 'delivery', 'cab'],
    'correct_choice_idx': 3,
    'direct_answers': ['ride', 'ride', 'bus', 'taxi', 'travelling', 'traffic', 'taxi', 'cab', 'cab', 'his ride'],
    'difficult_direct_answer': False,
    'rationales': ['A train would not be on the street, he would not have luggage waiting for a delivery,\
      and the skateboarder is there and not paying attention to him so a cab is the only possible answer.',
        'He has bags as if he is going someone, and he is on a road waiting for vehicle that can only be moved on the road and is big enough to hold the bags.',
    'He looks to be waiting for a paid ride to pick him up.'], 'instance_id': '0'}
    """
    # pdb.set_trace()
    ds_engine = deepspeed.init_inference(model,
                                    mp_size=1,
                                    dtype=torch.float16,
                                    #  checkpoint=None if args.pre_load_checkpoint else args.checkpoint_json,
                                    replace_with_kernel_inject=True 
                                    )
    model = ds_engine.module
    base_model = model.base_model.model
    # pdb.set_trace()
    dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=dataset.collater)
    batch_processor = FlamingoBatchProcessor(tokenizer=tokenizer, cast_type=torch.float16)
    with torch.no_grad():
        for data in dataloader:
            # for i in range(len(data['instruction'])):
            #     data['instruction'][i] = "<|endofchunk|><image>" + data['instruction'][i]
            # print(data['instruction'][0])                    # 
            # print(tokenizer.decode(data['input_ids'][0]))    # Q + A 
            # labels = data['labels']
            # label = data['labels'][0]
            # label = label[label > 0]    # [-100] padding
            # print(tokenizer.decode(label))
            # TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            # pdb.set_trace()
            output = batch_processor(model, data, mode='test')
            pdb.set_trace()
"""

data['instruction'][0]:
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Image:
<image>

### Instruction:
What is the occupation of the person driving? And can you tell me why?

### Response:

-----------------------------------------------------------------------------------------------------------
data['answer'][0]:
'The answer is farmer. Because The place is full of sheep that shows the person is a farmer. 
Farmer is the obvious profession as the picture shows.
 With the tractor he is in and the livestock shown it is easy to surmise his profession.'
-----------------------------------------------------------------------------------------------------------
output: 
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Image:


### Instruction:
What is the occupation of the person driving? And can you tell me why?

### Response:
The answer is farmer. Because The place is full of sheep that shows the person is a farmer. 
Farmer is the obvious profession as the picture shows.
With the tractor he is in and the livestock shown it is easy to surmise his profession.Home » Posts Tagged with: "Mongolia"
Mongolia’s nomadic

"""