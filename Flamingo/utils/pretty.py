
""" 常用函数 """
import json 
from termcolor import cprint
from PIL import Image
import numpy as np
from typing import List, Optional


def imread(path):
    """ 
        读取图片
    """
    return np.array(Image.open(path))

def pretty_print(*data_list, color="yellow"):
    """ 
        漂亮的打印
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