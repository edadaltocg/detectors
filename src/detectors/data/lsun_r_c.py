import os
import subprocess
from typing import Callable, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, extract_archive, verify_str_arg


class LSUNResized(ImageFolder):
    base_folder = "LSUN_resized"
    images_folder = ""
    filename = "LSUN_resize.tar.gz"
    file_md5 = "278b7b31c8cb7e804a1465a8ce03a2dc"
    url = "https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz"
    splits = ("all",)

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        self.root = os.path.expanduser(root)
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")
        super().__init__(
            self.split_folder,
            transform=transform,
            target_transform=target_transform,
            **kwargs,
        )

    @property
    def dataset_folder(self):
        return os.path.join(self.root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.images_folder)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.file_md5
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def _check_exists(self) -> bool:
        return os.path.exists(self.split_folder)

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            return
        print(f"Downloading {self.filename}...")
        archive = os.path.expanduser(os.path.join(self.root, self.filename))
        subprocess.run(
            f"wget {self.url} -P {self.root}".split(" "),
            capture_output=True,
            text=True,
        )
        extract_archive(archive, self.dataset_folder, False)


class LSUNCroped(ImageFolder):
    base_folder = "LSUN_croped"
    images_folder = ""
    filename = "LSUN.tar.gz"
    file_md5 = "458a0a0ab8e5f1cb4516d7400568e460"
    url = "https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz"
    splits = ("all",)

    def __init__(
        self,
        root: str,
        split: str = "all",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        self.root = os.path.expanduser(root)
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")
        super().__init__(
            self.split_folder,
            transform=transform,
            target_transform=target_transform,
            **kwargs,
        )

    @property
    def dataset_folder(self):
        return os.path.join(self.root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.images_folder)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.file_md5
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def _check_exists(self) -> bool:
        return os.path.exists(self.split_folder)

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            return
        print(f"Downloading {self.filename}...")
        archive = os.path.expanduser(os.path.join(self.root, self.filename))
        subprocess.run(
            f"wget {self.url} -P {self.root}".split(" "),
            capture_output=True,
            text=True,
        )
        extract_archive(archive, self.dataset_folder, False)
