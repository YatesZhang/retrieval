
""" 常用函数 """
import json 
from termcolor import cconsole.print
from PIL import Image
import numpy as np
from typing import List, Optional
from bigmodelvis import Visualization
from rich.console import Console
import torch 
console = Console()
def print_local_vars(func):
    """ 
        inline function can be replaced by decorator in python:
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        console.print("[bold magenta]Local [/bold magenta]!", ":vampire:", locals())
        return result
    return wrapper

def vis_model(model):
    """ 
        model visualization if rank == 0
    """
    try: 
        global_rank = torch.distributed.get_rank()
    except RuntimeError:
        model_struct = Visualization(model).structure_graph()
        return 
    if global_rank == 0:
        model_struct = Visualization(model).structure_graph()
        return model_struct 
    return 


def imread(path):
    """ 
        load image
    """
    return np.array(Image.open(path))

# @console.print_local_vars
def pretty_print(*data_list, color="yellow", line=False):
    """ 
        pretty print
    """
    if line:
        console.print("-------------------------------------------------", color=color)
    for data in data_list:  
        if isinstance(data, str):
            console.print(data, color)
        else:
            data = json.dumps(data, indent=2)
            data = data.replace("\"", "").replace(":"," =")
            console.print(data, color)
    return 



    