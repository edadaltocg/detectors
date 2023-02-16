from typing import Callable, Optional

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.utils import verify_str_arg


class CIFAR10Wrapped(CIFAR10):
    splits = ("train", "test")

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        self.split = verify_str_arg(split, "split", self.splits)

        super().__init__(
            root,
            train=split == "train",
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class CIFAR100Wrapped(CIFAR100):
    splits = ("train", "test")

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        self.split = verify_str_arg(split, "split", self.splits)

        super().__init__(
            root,
            train=split == "train",
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
