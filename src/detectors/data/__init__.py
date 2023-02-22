from typing import Optional, Type

import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import STL10, SVHN, ImageNet, Places365, StanfordCars

from detectors.data.cifar_wrapper import CIFAR10Wrapped, CIFAR100Wrapped
from detectors.data.cifarc import CIFAR10_C, CIFAR100_C
from detectors.data.imagenet_o import ImageNet_O
from detectors.data.mnist_wrapped import FashionMNISTWrapped, MNISTWrapped
from detectors.data.mos_inaturalist import MOSiNaturalist
from detectors.data.mos_places365 import MOSPlaces365
from detectors.data.mos_sun import MOSSUN

from ..config import DATA_DIR, IMAGENET_ROOT
from .constants import *
from .english_chars import EnglishChars
from .isun import iSUN
from .lsun_r_c import LSUNCroped, LSUNResized
from .noise import Gaussian, Uniform
from .textures import Textures
from .tiny_imagenet import TinyImageNet
from .tiny_imagenet_r_c import TinyImageNetCroped, TinyImageNetResized

datasets_registry = {
    "cifar10": CIFAR10Wrapped,
    "cifar100": CIFAR100Wrapped,
    "stl10": STL10,
    "svhn": SVHN,
    "mnist": MNISTWrapped,
    "fashion_mnist": FashionMNISTWrapped,
    "kmnist": ...,
    "emnist": ...,
    "english_chars": EnglishChars,
    "isun": iSUN,
    "lsun_c": LSUNCroped,
    "lsun_r": LSUNResized,
    "tiny_imagenet_c": TinyImageNetCroped,
    "tiny_imagenet_r": TinyImageNetResized,
    "tiny_imagenet": TinyImageNet,
    "textures": Textures,
    "gaussian": Gaussian,
    "uniform": Uniform,
    "places365": Places365,
    "stanford_cars": StanfordCars,
    "imagenet": ImageNet,
    "imagenet1k": ImageNet,
    "ilsvrc2012": ImageNet,
    "mos_inaturalist": MOSiNaturalist,
    "mos_places365": MOSPlaces365,
    "mos_sun": MOSSUN,
    "imagenet_o": ImageNet_O,
    "cifar10lt": ...,
    "cifar100lt": ...,
    "imagenet1klt": ...,
    "cifar10c": CIFAR10_C,
    "cifar100c": CIFAR100_C,
    "imagenet1kc": ...,
}


def register_dataset(dataset_name: str):
    def register_model_cls(cls):
        if dataset_name in datasets_registry:
            raise ValueError(f"Cannot register duplicate dataset ({dataset_name})")
        datasets_registry[dataset_name] = cls
        return cls

    return register_model_cls


def create_dataset(
    dataset_name: str,
    root: str = DATA_DIR,
    split: Optional[str] = "train",
    transform=None,
    download: Optional[bool] = True,
    **kwargs,
):
    try:
        if dataset_name in ["imagenet", "imagenet1k", "ilsvrc2012"]:
            return datasets_registry[dataset_name](root=IMAGENET_ROOT, split=split, transform=transform, **kwargs)
        return datasets_registry[dataset_name](root=root, split=split, transform=transform, download=download, **kwargs)
    except KeyError:
        raise ValueError("Dataset name is not specified")


def get_dataset_cls(dataset_name: str) -> Type[Dataset]:
    return datasets_registry[dataset_name]


def get_datasets_names():
    return list(datasets_registry.keys())
