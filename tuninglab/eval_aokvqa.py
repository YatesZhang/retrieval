"""
    Eval A-OK VQA test
    #TODO:
    implemnet this evaluation
    modify from:
    https://github.com/allenai/aokvqa/tree/main/evaluation
"""

import os
import json
from Flamingo.utils import pretty_print
import pdb 

def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    """ 
        load data
    """
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir,
         "aokvqa_{version}_{split}.json".format(version=version, split=split)))
    )
    return dataset

def get_coco_path(split, image_id, coco_dir):
    """ 
        coco path
    """
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")

if __name__ == "__main__":
    aokvqa_dir = "/home/yunzhi/datasets/aokvqa_v1p0"
    coco_dir = "/home/yunzhi/datasets/COCO"
    train_dataset = load_aokvqa(aokvqa_dir=aokvqa_dir, split='train')    # type: list

    dataset_example = train_dataset[0]

    print("dataset_example['question_id']: ",dataset_example['question_id'])
    # 22MexNkBPpdZGX6sxbxVBH

    image_path = get_coco_path('train', dataset_example['image_id'], coco_dir)
    print("image_path: ", image_path)
    # ./datasets/coco/train2017/000000299207.jpg

    print("dataset_example['question']: ", dataset_example['question'])
    print("dataset_example['choices']: ", dataset_example['choices'])
    print("dataset_example['direct_answers']", dataset_example['direct_answers'])
    # What is the man by the bags awaiting?
    # ['skateboarder', 'train', 'delivery', 'cab']

    correct_choice = dataset_example['choices'][ dataset_example['correct_choice_idx'] ]
    print("correct_choice: ",correct_choice)
    # Corrrect: cab

    print("dataset_example['rationales'][0]: ", dataset_example['rationales'][0])
    pretty_print(line=True)
    """ 
    Please prepare predictions_{split}.json files (for split: {val,test}) in the format below. 
    You may omit either multiple_choice or direct_answer field if you only want to evaluate one setting.
        {
            '<question_id>' : {
                'multiple_choice' : '<prediction>',
                'direct_answer' : '<prediction>'
            }
        }
    """
    pdb.set_trace()