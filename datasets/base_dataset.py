# partially borrowed from K-planes (https://github.com/sarafridov/K-Planes/blob/main/plenoxels/datasets/base_dataset.py)

from abc import ABC
import os
from typing import Optional, List, Union

import torch
from torch.utils.data import Dataset

from .intrinsics import Intrinsics


class BaseDataset(Dataset, ABC):
    def __init__(self,
                 datadir: str,
                 split: str,
                 rays_o: Optional[torch.Tensor],
                 rays_d: Optional[torch.Tensor],
                 intrinsics: Union[Intrinsics, List[Intrinsics]],
                 batch_size: Optional[int] = None,
                 imgs: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                 use_permutation: bool = True
                 ):
        self.datadir = datadir
        self.name = os.path.basename(self.datadir)
        self.split = split
        self.batch_size = batch_size
        if self.split == 'train':
            assert self.batch_size is not None
        self.rays_o = rays_o
        self.rays_d = rays_d
        self.imgs = imgs
        if self.imgs is not None:
            self.num_samples = len(self.imgs)
        elif self.rays_o is not None:
            self.num_samples = len(self.rays_o)
        else:
            self.num_samples = None
        self.intrinsics = intrinsics
        self.use_permutation = use_permutation
        self.perm = None

    @property
    def img_h(self) -> Union[int, List[int]]:
        if isinstance(self.intrinsics, list):
            return [i.height for i in self.intrinsics]
        return self.intrinsics.height

    @property
    def img_w(self) -> Union[int, List[int]]:
        if isinstance(self.intrinsics, list):
            return [i.width for i in self.intrinsics]
        return self.intrinsics.width

    def reset_iter(self):
        if self.use_permutation:
            self.perm = torch.randperm(self.num_samples)

    def get_rand_ids(self, index):
        assert self.batch_size is not None, "Can't get rand_ids for test split"
        if self.use_permutation:
            return self.perm[index * self.batch_size: (index + 1) * self.batch_size]
        else:
            return torch.randint(0, self.num_samples, size=(self.batch_size, ))

    def __len__(self):
        if self.split == 'train':
            return (self.num_samples + self.batch_size - 1) // self.batch_size
        else:
            return self.num_samples

    def __getitem__(self, index, return_idxs: bool = False):
        if self.split == 'train':
            index = self.get_rand_ids(index)
        out = {}
        if self.rays_o is not None:
            out["rays_o"] = self.rays_o[index]
        if self.rays_d is not None:
            out["rays_d"] = self.rays_d[index]
        if self.imgs is not None:
            out["imgs"] = self.imgs[index]
        else:
            out["imgs"] = None
        if return_idxs:
            return out, index
        return out
