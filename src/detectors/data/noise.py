from typing import Callable, Optional

import numpy as np
import torch
import torch.utils.data
from numpy.random import RandomState
from PIL import Image


class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support for transforms."""

    def __init__(self, *tensors, transform=None) -> None:
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform is not None:
            x = self.transform(self.tensors[0][index])

        return (x,) + tuple(tensor[index] for tensor in self.tensors[1:])

    def __len__(self):
        return len(self.tensors[0])


class Gaussian(CustomTensorDataset):
    def __init__(
        self,
        root: Optional[str] = None,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        nb_samples=10000,
        shape=(32, 32, 3),
        seed=1,
        **kwargs,
    ):
        rng = RandomState(seed)
        imgs = np.array(np.clip(rng.randn(nb_samples, *shape) + 0.5, 0, 1) * 255, dtype=np.uint8)
        labels = torch.tensor([0] * nb_samples)
        super().__init__(imgs, labels, transform=transform)


class Uniform(CustomTensorDataset):
    def __init__(
        self,
        root: Optional[str] = None,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        nb_samples=10000,
        shape=(32, 32, 3),
        seed=1,
        **kwargs,
    ):
        rng = RandomState(seed)
        imgs = np.array(rng.rand(nb_samples, *shape) * 255, dtype=np.uint8)
        labels = torch.tensor([0] * nb_samples)
        super().__init__(imgs, labels, transform=transform)
