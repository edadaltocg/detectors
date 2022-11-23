from typing import Callable, Optional

from torchvision import transforms
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


STATISTICS = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def default_cifar10_train_transforms():
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(*STATISTICS),
        ]
    )

    return test_transforms


def default_cifar10_test_transform():
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(*STATISTICS),
        ]
    )

    return test_transforms
