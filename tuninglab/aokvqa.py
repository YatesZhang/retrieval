"""
    dataset test
    test: 
        A-OK VQA
"""

from Flamingo.datasets.builder import build_dataset 
from Flamingo.config.baseline import dataset_config, model_config
from Flamingo.utils.pretty import pretty_print 
from Flamingo.lora_tuning import create_model_and_transforms
import pdb 
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
    _, image_processor, tokenizer = create_model_and_transforms(
        **model_config
    )
    # pdb.set_trace()
    dataset = build_dataset(
        dataset_config=dataset_config,
        vis_processor=image_processor,
        tokenizer=tokenizer,
    )
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
    for data in dataset:
        labels = data['labels']
        instruction = data['instruction']
        input_ids = data['input_ids']
        attentions_mask = data['attention_mask']
        image = data['image']
        answer = data['answer']
        pdb.set_trace()