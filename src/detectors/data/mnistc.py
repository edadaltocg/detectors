import logging
import os
from os.path import join
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

log = logging.getLogger(__name__)


class MNISTC:
    """
    MNIST-C is MNIST with corruptions for benchmarking OOD methods.

    Split can be one of `train`, `test` and `leftovers`.

    Subsets can be one of:
        `all`, `brightness`, `canny_edges`, `dotted_line`, `fog`, `glass_blur`, `identity`, `impulse_noise`,
        `motion_blur`, `rotate`, `scale`, `shear`, `shot_noise`, `spatter`, `stripe`, `translate` and `zigzag`.
    """

    splits = ["train", "test", "leftovers"]

    subsets = [
        "brightness",
        "canny_edges",
        "dotted_line",
        "fog",
        "glass_blur",
        "identity",
        "impulse_noise",
        "motion_blur",
        "rotate",
        "scale",
        "shear",
        "shot_noise",
        "spatter",
        "stripe",
        "translate",
        "zigzag",
    ]

    base_folders = ["mnist_c", "mnist_c_leftovers"]

    urls = [
        "https://zenodo.org/record/3239543/files/mnist_c.zip",
        "https://zenodo.org/record/3239543/files/mnist_c_leftovers.zip",
    ]

    filenames = [
        "mnist_c.zip",
        "mnist_c_leftovers.zip",
    ]
    tgz_md5s = [
        "4b34b33045869ee6d424616cd3a65da3",
        "c365e9c25addd5c24454b19ac7101070",
    ]

    def __init__(
        self,
        root: str,
        corruption: str,
        split: str,
        transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        self.root = root
        self.transform = transform

        if corruption not in self.subsets and corruption != "all":
            raise ValueError()

        if split not in self.splits:
            raise ValueError()

        self.base_folder = join(root, self.base_folders[1] if split == "leftovers" else self.base_folders[0])
        self.url = self.urls[0] if split in ["train", "test"] else self.urls[1]
        self.filename = self.filenames[0] if split in ["train", "test"] else self.filenames[1]
        self.tgz_md5 = self.tgz_md5s[0] if split in ["train", "test"] else self.tgz_md5s[1]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")

        self.subset = corruption

        if split == "leftovers":
            # TODO
            pass
        if corruption == "all":
            self.data = np.concatenate(
                [np.load(join(self.base_folder, s, f"{split}_images.npy")) for s in self.subsets]
            )
            self.targets = np.concatenate(
                [np.load(join(self.base_folder, s, f"{split}_labels.npy")) for s in self.subsets]
            )
        else:
            self.data = np.load(join(self.base_folder, corruption, f"{split}_images.npy"))
            self.targets = np.load(join(self.base_folder, corruption, f"{split}_labels.npy"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.data[index]
        target = self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze(), "L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.root, self.base_folder))

    def _check_integrity(self) -> bool:
        return check_integrity(join(self.root, self.filename), self.tgz_md5)

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            log.debug("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
