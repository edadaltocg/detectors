from typing import Callable, Optional

from torchvision.datasets import MNIST, FashionMNIST
from torchvision.datasets.utils import verify_str_arg


class MNISTWrapped(MNIST):
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


class FashionMNISTWrapped(FashionMNIST):
    splits = ("train", "test")

    def __init__(
        self,
        root: str,
        split: str = "all",
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


def test():
    dataset = MNISTWrapped("data", split="train", download=True)
    print(len(dataset))

    dataset = FashionMNISTWrapped("data", split="train", download=True)
    print(len(dataset))


if __name__ == "__main__":
    test()
