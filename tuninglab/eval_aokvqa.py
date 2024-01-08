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
from torch.utils.data import Dataset
from PIL import Image 
from rich import print 


class VQADataset(Dataset):
    def __init__(
        self, image_dir_path, question_path, annotations_path, is_train, dataset_name
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        if annotations_path is not None:
            self.answers = json.load(open(annotations_path, "r"))["annotations"]
        else:
            self.answers = None
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            self.img_coco_split = self.image_dir_path.strip("/").split("/")[-1]
            assert self.img_coco_split in {"train2014", "val2014", "test2015"}

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            return os.path.join(
                self.image_dir_path,
                f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.image_dir_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.image_dir_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        results = {
            "image": image,
            "question": question["question"],
            "question_id": question["question_id"],
        }
        if self.answers is not None:
            answers = self.answers[idx]
            results["answers"] = [a["answer"] for a in answers["answers"]]
        return results
    

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
    