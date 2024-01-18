""" 
    dataset build 
    copy from mmgpt
    https://github.com/open-mmlab/Multimodal-GPT/tree/main/mmgpt
"""

import numpy as np
import torch
from .vqa_dataset import MyConcatDataset, VQADataset  # noqa: F401
from .gtsrb import GTSRB

def build_dataset(dataset_config, vis_processor, tokenizer):
    """ 
        build dataset
    """
    if isinstance(dataset_config, list):
        datasets = [build_dataset(cfg, vis_processor=vis_processor, tokenizer=tokenizer) for cfg in dataset_config]
        return MyConcatDataset(datasets)
    _dataset_config = dataset_config.copy()
    dataset_type = _dataset_config.pop("type")
    sample = _dataset_config.pop("sample", -1)
    if dataset_type == 'gtsrb':
        dataset = GTSRB(
            tokenizer=tokenizer, 
            **_dataset_config             
        )
    elif dataset_type == "vqa":
        dataset = VQADataset(
            vis_processor=vis_processor,
            tokenizer=tokenizer,
            **_dataset_config
        )
    elif dataset_type == "participants_property":
        dataset = ParticipantsPropertyDataset(
            **_dataset_config
        )
    else:
        raise NotImplementedError

    if sample > 0:
        random_indices = np.random.choice(len(dataset), min(sample, len(dataset)), replace=False)
        subsample_dataset = torch.utils.data.Subset(dataset, random_indices)
        subsample_dataset.collater = dataset.collater
        return subsample_dataset
    else:
        return dataset
