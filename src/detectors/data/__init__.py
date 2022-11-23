from typing import Optional, Type

from detectors.data.cifar_wrapper import CIFAR10Wrapped, CIFAR100Wrapped
from detectors.data.imagenet_o import ImageNet_O
from detectors.data.mnist_wrapped import FashionMNISTWrapped, MNISTWrapped
from detectors.data.mos_inaturalist import MOSiNaturalist
from detectors.data.mos_places365 import MOSPlaces365
from detectors.data.mos_sun import MOSSUN
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import STL10, SVHN, ImageNet, StanfordCars

from ..config import DATASETS_DIR, IMAGENET_ROOT
from .english_chars import EnglishChars
from .isun import iSUN
from .lsun_r_c import LSUNCroped, LSUNResized
from .noise import Gaussian, Uniform
from .places365 import Places365
from .textures import Textures
from .tiny_imagenet import TinyImageNet
from .tiny_imagenet_r_c import TinyImageNetCroped, TinyImageNetResized


datasets_registry = {
    "cifar10": CIFAR10Wrapped,
    "cifar100": CIFAR100Wrapped,
    "svhn": SVHN,
    "mnist": MNISTWrapped,
    "fashion_minst": FashionMNISTWrapped,
    "english_chars": EnglishChars,
    "isun": iSUN,
    "lsun_c": LSUNCroped,
    "lsun_r": LSUNResized,
    "gaussian": Gaussian,
    "uniform": Uniform,
    "places365": Places365,
    "textures": Textures,
    "tiny_imagenet": TinyImageNet,
    "tiny_imagenet_c": TinyImageNetCroped,
    "tiny_imagenet_r": TinyImageNetResized,
    "stanford_cars": StanfordCars,
    "stl10": STL10,
    "imagenet1k": ImageNet,
    "ilsvrc2012": ImageNet,
    "imagenet_o": ImageNet_O,
    "mos_inaturalist": MOSiNaturalist,
    "mos_places365": MOSPlaces365,
    "mos_sun": MOSSUN,
}


def register_dataset(dataset_name: str):
    def register_model_cls(cls):
        if dataset_name in datasets_registry:
            raise ValueError(f"Cannot register duplicate dataset ({dataset_name})")
        datasets_registry[dataset_name] = cls
        return cls

    return register_model_cls


def get_dataset(dataset_name: Optional[str] = None, root: str = DATASETS_DIR, transform=None, **kwargs):
    if dataset_name is not None:
        if dataset_name in ["imagenet1k", "ilsvrc2012"]:
            root = IMAGENET_ROOT
            kwargs.pop("download", None)
        return datasets_registry[dataset_name](root=root, transform=transform, **kwargs)

    raise ValueError("Dataset name is not specified")


def get_dataset_cls(dataset_name: str) -> Type[Dataset]:
    return datasets_registry[dataset_name]


def get_datasets_names():
    return list(datasets_registry.keys())


def default_imagenet_test_transforms():
    statistics = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*statistics),
        ]
    )
    return test_transforms
