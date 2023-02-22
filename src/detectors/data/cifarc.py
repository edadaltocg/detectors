import os
from typing import Callable, Optional

import numpy as np
import torch.utils.data.dataset as dataset
from PIL import Image
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg

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


class CIFAR10_C(dataset.Dataset):
    base_folder = "CIFAR-10-C"
    filename = "CIFAR-10-C.tar"
    file_md5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    corruptions = CORRUPTIONS

    def __init__(
        self, root: str, split: str, intensity: int, transform: Optional[Callable] = None, download: bool = False
    ) -> None:
        self.root = os.path.expanduser(root)
        self.corruption = verify_str_arg(split, "split", self.corruptions)
        self.intensity = int(intensity)
        self.transform = transform
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")

        indices = self._indices
        images = np.load(self._perturbation_array)[indices]
        labels = np.load(self._labels_array)[indices]
        self.arrays = [images, labels]

    def __getitem__(self, index):
        x = self.arrays[0][index]

        # to PIL image
        x = Image.fromarray(x)

        if self.transform:
            x = self.transform(x)

        y = self.arrays[1][index]
        return x, y

    def __len__(self):
        return len(self.arrays[0])

    @property
    def _indices(self):
        N = 10000
        return slice((self.intensity - 1) * N, self.intensity * N)

    @property
    def _dataset_folder(self):
        return os.path.join(self.root, self.base_folder)

    @property
    def _perturbation_array(self):
        return os.path.join(self._dataset_folder, self.corruption + ".npy")

    @property
    def _labels_array(self):
        return os.path.join(self._dataset_folder, "labels.npy")

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.file_md5
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def _check_exists(self) -> bool:
        return os.path.exists(self._perturbation_array)

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            return
        download_and_extract_archive(self.url, download_root=self.root, md5=self.file_md5)


class CIFAR100_C(CIFAR10_C):
    base_folder = "CIFAR-100-C"
    filename = "CIFAR-100-C.tar"
    file_md5 = "11f0ed0f1191edbf9fa23466ae6021d3"
    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
    corruptions = CORRUPTIONS
