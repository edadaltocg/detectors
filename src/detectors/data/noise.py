from typing import Callable, Optional

import numpy as np
import torch
import torch.utils.data


class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support for transforms."""

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = int(self.tensors[1][index])
        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]


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
        **kwargs,
    ):
        tensors = [
            np.clip(np.random.randn(nb_samples, *shape) + 0.5, 0, 1),
            np.array([1.0] * nb_samples),
        ]
        super().__init__(tensors, transform)


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
    ):
        tensors = [
            np.random.rand(nb_samples, *shape),
            np.array([1.0] * nb_samples),
        ]
        super().__init__(tensors, transform)


def test():
    gaussian = Gaussian()
    print(gaussian)
    print(gaussian[0])
    print(len(gaussian))
    uniform = Uniform()
    print(uniform)
    print(uniform[0])
    print(len(uniform))


if __name__ == "__main__":
    test()
