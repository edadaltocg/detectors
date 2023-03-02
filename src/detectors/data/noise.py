from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from numpy.random import RandomState


class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support for transformations.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
        transform (callable, optional): transform to apply.
    """

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
    """Gaussian noise dataset.

    Args:
        root (str): root directory.
        split (str, optional): not used.
        transform (callable, optional): transform to apply.
        download (bool, optional): not used.
        nb_samples (int): number of samples.
        shape (tuple[int]): shape of the samples.
        seed (int): seed for the random number generator.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        download: bool = False,
        nb_samples=10000,
        shape: Tuple[int] = tuple([32, 32, 3]),
        seed=1,
        **kwargs,
    ):
        rng = RandomState(seed)
        imgs = np.array(np.clip(rng.randn(nb_samples, *shape) + 0.5, 0, 1) * 255, dtype=np.uint8)
        labels = torch.tensor([0] * nb_samples)
        super().__init__(imgs, labels, transform=transform)


class Uniform(CustomTensorDataset):
    """Uniform noise dataset.

    Args:
        root (str): root directory.
        split (str, optional): not used.
        transform (callable, optional): transform to apply.
        download (bool, optional): not used.
        nb_samples (int): number of samples.
        shape (tuple[int]): shape of the samples.
        seed (int): seed for the random number generator.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
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
