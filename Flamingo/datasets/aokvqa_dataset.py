"""
AOKVQADataset
"""

import random

from .vqa_dataset import VQADataset
import os 
from PIL import Image

REASON_QUESTIONS = [
    "Why?",
    "Why is this?",
    "And why?",
    "What is the reason?",
    "And can you tell me why?",
    "Can you tell me why?",
    "Can you tell me the reason?",
]


class AOKVQADataset(VQADataset):
    def __init__(self, tokenizer, vis_processor, vis_root, ann_paths, **kwargs):
        """ 
            init
        """
        # tokenizer.eos_token = None
        super().__init__(tokenizer, vis_processor, vis_root, ann_paths, **kwargs)

    def get_path(self, ann):
        """ 
            get path
        """
        image_id = str(ann["image_id"])
        while len(image_id) != 12:
            image_id = "0" + image_id
        image_id += ".jpg"
        image_id = os.path.join(self.vis_root, image_id)
        return image_id
    
    def process_image(self, ann):
        """ 
            process image
        """
        image_path = self.get_path(ann)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        return image
    
    def process_text(self, ann):
        """ 
            process_text
        """
        question = ann["question"]
        question = question + " " + random.choice(REASON_QUESTIONS)

        choices = ann["choices"]
        true_answer = choices[ann["correct_choice_idx"]]
        answer = "The answer is " + true_answer + ". Because " + " ".join(ann["rationales"])

        is_option = random.random() < self.option_prob and len(choices) > 1
        if is_option:
            instruction = self.prompter(question, choices)
        else:
            instruction = self.prompter(question)

        instruction = self.prompter(question)
        return dict(instruction=instruction, answer=answer)


def build_aokvqa_dataset(
    tokenizer,
    vis_processor,
    vis_root="data/coco/images",
    ann_paths=["data/aokvqa/annotations/aokvqa_v1p0_train.json"],
    sample_image=False,
):
    """ 
        build dataset
    """
    return AOKVQADataset(
        tokenizer=tokenizer,
        vis_processor=vis_processor,
        vis_root=vis_root,
        ann_paths=ann_paths,
        sample_image=sample_image,
    )
