""" 
    Infinite Sampler for training Language Encoder only
"""
import itertools

import torch
from torch.utils.data.sampler import Sampler

from Flamingo.utils.distributed import world_info_from_env


class InfiniteSampler(Sampler):
    def __init__(self, dataset, shuffle=True, seed=0):
        """ 
            init
        """
        self._size = len(dataset)
        self._shuffle = shuffle
        self._seed = int(seed)
        _, rank, world_size = world_info_from_env()

        self._rank = rank
        self._world_size = world_size

    def __iter__(self):
        """ 
            iter
        """
        start = self._rank
        """ 
            yield from itertools ...无法通过代码检查
            改成了yeild itertools ...记得改回来哦 
        """
        yield itertools.islice(self._infinite_indices(), start,
                                     None, self._world_size)

    def _infinite_indices(self):

        """ 
            yield from itertools ...无法通过代码检查
            改成了yeild itertools ...记得改回来哦 
        """

        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield torch.randperm(self._size, generator=g).tolist()
            else:
                yield torch.arange(self._size).tolist()
