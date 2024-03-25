
""" 常用函数 """
import json 
from termcolor import cprint
from PIL import Image
import numpy as np
from typing import List, Optional
from bigmodelvis import Visualization
from rich.console import Console
import torch 
console = Console()

def print_local_vars(func):
    raise NotImplementedError
    # """ 
    #     inline function can be replaced by decorator in python:
    # """
    # def wrapper(*args, **kwargs):
    #     result = func(*args, **kwargs)
    #     console.print("[bold magenta]Local [/bold magenta]!", ":vampire:", locals())
    #     return result
    # return wrapper

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
def pretty_print(data_list, color="yellow", line=False):
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


def show_anns(anns):
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(anns[0].shape[0], anns[0].shape[1], 4)
    img[:,:,3] = 0
    for ann in anns:
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[ann] = color_mask
    ax.imshow(img)


def show_masks(img, masks):
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show() 
    