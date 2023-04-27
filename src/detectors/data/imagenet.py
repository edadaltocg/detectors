import logging
import os
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg
from tqdm import tqdm

_logger = logging.getLogger(__name__)


class ImageNetA(ImageFolder):
    """ImageNetA dataset.

    - Paper: [https://arxiv.org/abs/1907.07174](https://arxiv.org/abs/1907.07174).
    """

    base_folder = "imagenet-a"
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
    filename = "imagenet-a.tar"
    tgz_md5 = "c3e55429088dc681f30d81f4726b6595"

    def __init__(self, root: str, split=None, transform: Optional[Callable] = None, download: bool = False, **kwargs):

        self.root = root

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")

        super().__init__(root=os.path.join(root, self.base_folder), transform=transform, **kwargs)

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.root, self.base_folder))

    def _check_integrity(self) -> bool:
        return check_integrity(os.path.join(self.root, self.filename), self.tgz_md5)

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            _logger.debug("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)


class ImageNetO(ImageNetA):
    """ImageNetO datasets.

    Contains unknown classes to ImageNet-1k.


    - Paper: [https://arxiv.org/abs/1907.07174](https://arxiv.org/abs/1907.07174)
    """

    base_folder = "imagenet-o"
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar"
    filename = "imagenet-o.tar"
    tgz_md5 = "86bd7a50c1c4074fb18fc5f219d6d50b"


class ImageNetR(ImageNetA):
    """ImageNet-R(endition) dataset.

    Contains art, cartoons, deviantart, graffiti, embroidery, graphics, origami, paintings,
    patterns, plastic objects,plush objects, sculptures, sketches, tattoos, toys,
    and video game renditions of ImageNet-1k classes.

    - Paper: [https://arxiv.org/abs/2006.16241](https://arxiv.org/abs/2006.16241)
    """

    base_folder = "imagenet-r"
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    filename = "imagenet-r.tar"
    tgz_md5 = "a61312130a589d0ca1a8fca1f2bd3337"


CORRUPTIONS = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]


class ImageNetC(ImageNetA):
    """Corrupted version of the ImageNet-1k dataset.

    It contains the following subsets:

    - `noise` (21GB): gaussian_noise, shot_noise, and impulse_noise.
    - `blur` (7GB): defocus_blur, glass_blur, motion_blur, and zoom_blur.
    - `weather` (12GB):  frost, snow, fog, and brightness.
    - `digital` (7GB): contrast, elastic_transform, pixelate, and jpeg_compression.
    - `extra` (15GB): speckle_noise, spatter, gaussian_blur, and saturate.

    - Paper: [https://arxiv.org/abs/1903.12261v1](https://arxiv.org/abs/1903.12261v1)
    """

    split_list = ["blur", "digital", "extra", "noise", "weather"]
    base_folder_name = "ImageNetC"
    url_base = "https://zenodo.org/record/2235448/files/"
    tgz_md5_list = [
        "2d8e81fdd8e07fef67b9334fa635e45c",
        "89157860d7b10d5797849337ca2e5c03",
        "d492dfba5fc162d8ec2c3cd8ee672984",
        "e80562d7f6c3f8834afb1ecf27252745",
        "33ffea4db4d93fe4a428c40a6ce0c25d",
    ]
    corruptions = CORRUPTIONS

    def __init__(
        self,
        root: str,
        split: str,
        intensity: int,
        transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        self.root = os.path.expanduser(root)
        self.corruption = verify_str_arg(split, "split", self.corruptions)
        split_group = self._get_corruption_group(self.corruption)

        self._base_folder = os.path.join(root, self.base_folder_name, split_group)
        self.filename = split_group + ".tar"
        self.url = self.url_base + self.filename
        self.tgz_md5 = self.tgz_md5_list[self.split_list.index(split_group)]

        self.base_folder = os.path.join(self._base_folder, split, str(intensity))

        super().__init__(root, transform=transform, download=download, **kwargs)

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            _logger.debug("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self.root, extract_root=self._base_folder, filename=self.filename, md5=self.tgz_md5
        )

    @staticmethod
    def _get_corruption_group(corruption: str):
        split_group = ""
        if corruption in ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"]:
            split_group = "blur"
        elif corruption in ["contrast", "elastic_transform", "pixelate", "jpeg_compression"]:
            split_group = "digital"
        elif corruption in ["speckle_noise", "spatter", "gaussian_blur", "saturate"]:
            split_group = "extra"
        elif corruption in ["gaussian_noise", "shot_noise", "impulse_noise"]:
            split_group = "noise"
        elif corruption in ["frost", "snow", "fog", "brightness"]:
            split_group = "weather"
        return split_group


def _imagenet_c_to_npz(root: str, split: str, intensity: int, dest_folder: str = "ImageNetCnpz") -> None:

    dataset = ImageNetC(root, split, intensity, download=True)
    assert len(dataset) == 50_000, "ImageNetC should have 50,000 images. Please check the dataset."
    image_example = dataset[0][0]
    width, height = image_example.size
    _logger.info("Image size: %d x %d", width, height)
    x = np.ndarray(shape=(len(dataset), height, width, 3), dtype=np.uint8)
    y = np.ndarray(shape=(len(dataset)), dtype=np.int32)
    for i in tqdm(range(len(dataset))):
        image, label = dataset[i]
        x[i] = image
        y[i] = label

    os.makedirs(os.path.join(root, dest_folder), exist_ok=True)
    np.savez(os.path.join(root, dest_folder, f"{split}-{intensity}.npz"), x=x, y=y)


class ImageNetCnpz(Dataset):
    """Corrupted version of the ImageNet-1k dataset saved in npz format."""

    corruptions = CORRUPTIONS
    base_folder_name = "ImageNetCnpz"

    def __init__(
        self,
        root: str,
        split: str,
        intensity: int,
        transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = os.path.expanduser(root)
        self.corruption = verify_str_arg(split, "split", self.corruptions)
        self.intensity = int(intensity)
        self.path = os.path.join(self.root, self.base_folder_name, f"{split}-{intensity}.npz")
        self.transform = transform
        if download:
            self.download()

        data = np.load(self.path, mmap_mode="r")
        self.images = data["x"]
        self.labels = data["y"]

    def __getitem__(self, index):
        x = self.images[index]
        x = Image.fromarray(x)

        if self.transform:
            x = self.transform(x)

        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.images)

    def _check_exists(self) -> bool:
        return os.path.exists(self.path)

    def download(self) -> None:
        if self._check_exists():
            return
        _imagenet_c_to_npz(self.root, self.corruption, self.intensity, self.base_folder_name)
