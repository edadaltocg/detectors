import os
from typing import Any, Callable, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg


class ImageNet_O(ImageFolder):
    base_folder = "imagenet-o"
    images_folder = "imagenet-o"
    filename = "imagenet-o.tar"
    file_md5 = "86bd7a50c1c4074fb18fc5f219d6d50b"
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar"
    splits = ("all",)

    def __init__(
        self,
        root: str,
        split: str = "all",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.root = os.path.expanduser(root)

        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")
        super().__init__(
            self._split_folder,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

    @property
    def _dataset_folder(self):
        return os.path.join(self.root, self.base_folder)

    @property
    def _split_folder(self):
        return os.path.join(self._dataset_folder)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.file_md5
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def _check_exists(self) -> bool:
        return os.path.exists(self._split_folder)

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            return
        download_and_extract_archive(
            self.url,
            download_root=self.root,
            extract_root=self._dataset_folder,
            remove_finished=False,
            md5=self.file_md5,
        )
