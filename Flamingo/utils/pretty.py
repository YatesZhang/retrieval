
""" 常用函数 """
import json 
from termcolor import cprint
from PIL import Image
import numpy as np
from typing import List, Optional
from bigmodelvis import Visualization

def vis_model(model):
    """ 
        model visualization
    """
    Visualization(model).structure_graph()
    return 


def imread(path):
    """ 
        load image
    """
    return np.array(Image.open(path))


def pretty_print(*data_list, color="yellow"):
    """ 
        pretty print
    """
    for data in data_list:  
        if isinstance(data, str):
            cprint(data, color)
        else:
            data = json.dumps(data, indent=2)
            data = data.replace("\"", "").replace(":"," =")
            cprint(data, color)
    return 

# if __name__ == "__main__":
#     test_dict = dict(
#         a=1,
#         b=2,
#         c="hello"
#     )
#     pretty_print(test_dict)