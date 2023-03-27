"""Data module."""
from functools import partial
import logging
from typing import Callable, Optional, Type

from torch.utils.data import Dataset
from torchvision.datasets import STL10, SVHN, ImageNet, OxfordIIITPet, StanfordCars

from detectors.data.cifar_wrapper import CIFAR10Wrapped, CIFAR100Wrapped
from detectors.data.cifarc import CIFAR10_C, CIFAR100_C
from detectors.data.imagenet import ImageNetA, ImageNetC, ImageNetO, ImageNetR
from detectors.data.mnist_wrapped import FashionMNISTWrapped, MNISTWrapped
from detectors.data.mnistc import MNISTC
from detectors.data.mos import MOSSUN, MOSiNaturalist, MOSPlaces365
from detectors.data.openimage_o import OpenImageO
from detectors.data.places365 import Places365
from detectors.data.wilds_ds import make_wilds_dataset

from ..config import DATA_DIR, IMAGENET_ROOT
from .constants import *
from .english_chars import EnglishChars
from .isun import iSUN
from .lsun_r_c import LSUNCroped, LSUNResized
from .noise import Gaussian, Uniform
from .textures import Textures
from .tiny_imagenet import TinyImageNet
from .tiny_imagenet_r_c import TinyImageNetCroped, TinyImageNetResized

_logger = logging.getLogger(__name__)
datasets_registry = {
    "cifar10": CIFAR10Wrapped,
    "cifar100": CIFAR100Wrapped,
    "stl10": STL10,
    "svhn": SVHN,
    "mnist": MNISTWrapped,
    "fashion_mnist": FashionMNISTWrapped,
    "kmnist": ...,
    "emnist": ...,
    "mnist_c": MNISTC,
    "english_chars": EnglishChars,
    "isun": iSUN,
    "lsun_c": LSUNCroped,
    "lsun_r": LSUNResized,
    "tiny_imagenet_c": TinyImageNetCroped,
    "tiny_imagenet_r": TinyImageNetResized,
    "tiny_imagenet": TinyImageNet,
    "textures": Textures,
    "textures_curated": ...,
    "gaussian": Gaussian,
    "uniform": Uniform,
    "places365": Places365,
    "stanford_cars": StanfordCars,
    "mos_inaturalist": MOSiNaturalist,
    "mos_places365": MOSPlaces365,
    "mos_sun": MOSSUN,
    "cifar10lt": ...,
    "cifar100lt": ...,
    "imagenet1klt": ...,
    "cifar10c": CIFAR10_C,
    "cifar100c": CIFAR100_C,
    "imagenet": ImageNet,
    "imagenet1k": ImageNet,
    "ilsvrc2012": ImageNet,
    "imagenetc": ImageNetC,
    "imagenet1kc": ImageNetC,
    "imagenet_a": ImageNetA,
    "imagenet_r": ImageNetR,
    "imagenet_o": ImageNetO,
    "openimage_o": OpenImageO,
    "oxford_pets": OxfordIIITPet,
    "oxford_flowers": ...,
    "cub200": ...,
    "iwildcam": partial(make_wilds_dataset, dataset_name="iwildcam"),
    "fmow": partial(make_wilds_dataset, dataset_name="fmow"),
    "camelyon17": partial(make_wilds_dataset, dataset_name="camelyon17"),
    "rxrx1": partial(make_wilds_dataset, dataset_name="rxrx1"),
    "poverty": partial(make_wilds_dataset, dataset_name="poverty"),
    "globalwheat": partial(make_wilds_dataset, dataset_name="globalwheat"),
}


def register_dataset(dataset_name: str):
    """Register a dataset on the `datasets_registry`."""

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
    transform: Optional[Callable] = None,
    download: Optional[bool] = True,
    **kwargs,
):
    """Create dataset factory.

    Args:
        dataset_name (string): Name of the dataset.
            Already implemented:
            [`cifar10`, `cifar100`, `stl10`, `svhn`, `mnist`, `fashion_mnist`,
              `kmnist`, `emnist`, `mnist_c`, `english_chars`, `isun`, `lsun_c`, `lsun_r`,
              `tiny_imagenet_c`, `tiny_imagenet_r`, `tiny_imagenet`, `textures`, `gaussian`,
              `uniform`, `places365`, `stanford_cars`, `imagenet`, `imagenet1k`, `ilsvrc2012`,
              `mos_inaturalist`, `mos_places365`, `mos_sun`, `cifar10lt`, `cifar100lt`,
              `imagenet1klt`, `cifar10c`, `cifar100c`, `imagenet_c`, `imagenet_a`,
              `imagenet_r`, `imagenet_o`, `oxford_pets`, `oxford_flowers`, `cub200`]
        root (string): Root directory of dataset.
        split (string, optional): Depends on the selected dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    Raises:
        ValueError: If dataset name is not specified.

    Returns:
        Dataset: Dataset object.
    """
    try:
        if dataset_name in ["imagenet", "imagenet1k", "ilsvrc2012"]:
            return datasets_registry[dataset_name](root=IMAGENET_ROOT, split=split, transform=transform, **kwargs)
        return datasets_registry[dataset_name](root=root, split=split, transform=transform, download=download, **kwargs)
    except KeyError as e:
        _logger.error(e)
        raise ValueError("Dataset name is not specified")


def get_dataset_cls(dataset_name: str) -> Type[Dataset]:
    return datasets_registry[dataset_name]


def list_datasets():
    """Return list of available dataset names, sorted alphabetically"""
    return sorted(list(datasets_registry.keys()))
