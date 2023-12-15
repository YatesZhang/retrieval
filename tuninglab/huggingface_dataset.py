""" 
    tutorial: https://huggingface.co/docs/datasets/load_hub
"""

from datasets import load_dataset_builder
from datasets import load_dataset
from datasets import get_dataset_split_names
from datasets import get_dataset_config_names
import pdb
""" 
Use the load_dataset_builder() function to load a dataset builder and inspect a dataset’s attributes
 without committing to downloading it:
"""
name = "rotten_tomatoes"
name = "samsum"
ds_builder = load_dataset_builder(name)
""" 
Inspect dataset's info:
    ds_builder.info.description
    ds_builder.info.features

    get_dataset_split_names(name)
    get_dataset_config_names(name)

"""
if __name__ == "__main__":
    dataset = load_dataset(name, split="train")

    pdb.set_trace()