import os.path as osp 

def path_finder(PATHS: list):
    """
        auto find path in PATHS
    """
    for path in PATHS:
        if osp.exists(path):
            return path
    raise FileNotFoundError(f"path {path} not found")