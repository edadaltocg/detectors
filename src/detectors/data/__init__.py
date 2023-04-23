"""
Datasets module.
"""
import logging
from enum import Enum
from functools import partial
from typing import Callable, List, Optional, Type

from torch.utils.data import Dataset
from torchvision.datasets import STL10, SVHN, ImageNet, OxfordIIITPet, StanfordCars

from ..config import DATA_DIR, IMAGENET_ROOT
from .cifar_wrapper import CIFAR10Wrapped, CIFAR100Wrapped
from .cifarc import CIFAR10_C, CIFAR100_C
from .constants import *
from .english_chars import EnglishChars
from .imagenet import ImageNetA, ImageNetC, ImageNetO, ImageNetR
from .isun import iSUN
from .lsun_r_c import LSUNCroped, LSUNResized
from .mnist_wrapped import FashionMNISTWrapped, MNISTWrapped
from .mnistc import MNISTC
from .mos import MOSSUN, MOSiNaturalist, MOSPlaces365
from .noise import Blobs, Gaussian, Uniform
from .openimage_o import OpenImageO
from .places365 import Places365
from .textures import Textures
from .tiny_imagenet import TinyImageNet
from .tiny_imagenet_r_c import TinyImageNetCroped, TinyImageNetResized
from .wilds_ds import make_wilds_dataset

_logger = logging.getLogger(__name__)
datasets_registry = {
    "cifar10": CIFAR10Wrapped,
    "cifar100": CIFAR100Wrapped,
    "stl10": STL10,
    "svhn": SVHN,
    "mnist": MNISTWrapped,
    "fashion_mnist": FashionMNISTWrapped,
    "kmnist": None,
    "emnist": None,
    "mnist_c": MNISTC,
    "english_chars": EnglishChars,
    "isun": iSUN,
    "lsun_c": LSUNCroped,
    "lsun_r": LSUNResized,
    "tiny_imagenet_c": TinyImageNetCroped,
    "tiny_imagenet_r": TinyImageNetResized,
    "tiny_imagenet": TinyImageNet,
    "textures": Textures,
    "textures_curated": None,
    "gaussian": Gaussian,
    "uniform": Uniform,
    "blobs": Blobs,
    "places365": Places365,
    "stanford_cars": StanfordCars,
    "mos_inaturalist": MOSiNaturalist,
    "mos_places365": MOSPlaces365,
    "mos_sun": MOSSUN,
    "cifar10_lt": None,
    "cifar100_lt": None,
    "imagenet1k_lt": None,
    "cifar10_c": CIFAR10_C,
    "cifar100_c": CIFAR100_C,
    "imagenet": ImageNet,
    "imagenet1k": ImageNet,
    "ilsvrc2012": ImageNet,
    "imagenet_c": ImageNetC,
    "imagenet1k_c": ImageNetC,
    "imagenet_a": ImageNetA,
    "imagenet_r": ImageNetR,
    "imagenet_o": ImageNetO,
    "openimage_o": OpenImageO,
    "oxford_pets": OxfordIIITPet,
    "oxford_flowers": None,
    "cub200": None,
    "wilds_iwildcam": partial(make_wilds_dataset, dataset_name="iwildcam"),
    "wilds_fmow": partial(make_wilds_dataset, dataset_name="fmow"),
    "wilds_camelyon17": partial(make_wilds_dataset, dataset_name="camelyon17"),
    "wilds_rxrx1": partial(make_wilds_dataset, dataset_name="rxrx1"),
    "wilds_poverty": partial(make_wilds_dataset, dataset_name="poverty"),
    "wilds_globalwheat": partial(make_wilds_dataset, dataset_name="globalwheat"),
}


def register_dataset(dataset_name: str):
    """Register a dataset on the `datasets_registry`.

    Args:
        dataset_name (str): Name of the dataset.

    Example::

        @register_dataset("my_dataset")
        class MyDataset(Dataset):
            ...

        dataset = create_dataset("my_dataset")
    """

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
                `cifar10`, `cifar100`, `stl10`, `svhn`, `mnist`, `fashion_mnist`,
                `kmnist`, `emnist`, `mnist_c`, `english_chars`, `isun`, `lsun_c`, `lsun_r`,
                `tiny_imagenet_c`, `tiny_imagenet_r`, `tiny_imagenet`, `textures`, `gaussian`,
                `uniform`, `places365`, `stanford_cars`, `imagenet`, `imagenet1k`, `ilsvrc2012`,
                `mos_inaturalist`, `mos_places365`, `mos_sun`, `cifar10_lt`, `cifar100_lt`,
                `imagenet1k_lt`, `cifar10_c`, `cifar100_c`, `imagenet_c`, `imagenet_a`,
                `imagenet_r`, `imagenet_o`, `openimage_o`, `oxford_pets`, `oxford_flowers`,
                `cub200`, `imagenet1k_c`, `blobs`.
                `wilds_iwildcam`, `wilds_fmow`, `wilds_camelyon17`, `wilds_rxrx1`,
                `wilds_poverty`, `wilds_globalwheat`.
        root (string): Root directory of dataset.
        split (string, optional): Depends on the selected dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, `transforms.RandomCrop`
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        **kwargs: Additional arguments for dataset.

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
    """Return dataset class by name.

    Args:
        dataset_name (string): Name of the dataset.

    Raises:
        ValueError: If dataset name is not available in `datasets_registry`.

    Returns:
        Dataset: Dataset class.
    """
    return datasets_registry[dataset_name]


def list_datasets() -> List[str]:
    """List of available dataset names, sorted alphabetically.

    Returns:
        list: List of available dataset names.
    """
    return sorted(list(k for k in datasets_registry.keys() if datasets_registry[k] is not None))


DatasetsRegistry = Enum("DatasetsRegistry", dict(zip(list_datasets(), list_datasets())))
