
""" 常用函数 """
import json 
from termcolor import cprint
from PIL import Image
import numpy as np
from typing import List, Optional
from bigmodelvis import Visualization
from rich import print as rich_print
import torch 

def print_local_vars(func):
    """ 
        inline function can be replaced by decorator in python:
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        rich_print("[bold magenta]Local [/bold magenta]!", ":vampire:", locals())
        return result
    return wrapper


def vis_model(model):
    global_rank = torch.distributed.get_rank()
    """ 
        model visualization
    """
    if global_rank == 0:
        model_struct = Visualization(model).structure_graph()
        return model_struct 
    return 


def imread(path):
    """ 
        load image
    """
    return np.array(Image.open(path))

# @print_local_vars
def pretty_print(*data_list, color="yellow", line=False):
    """ 
        pretty print
    """
    if line:
        cprint("-------------------------------------------------", color=color)
    for data in data_list:  
        if isinstance(data, str):
            cprint(data, color)
        else:
            data = json.dumps(data, indent=2)
            data = data.replace("\"", "").replace(":"," =")
            cprint(data, color)
    return 



    