import os
from typing import Callable, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class Places365(ImageFolder):
    base_folder = "places365"
    filename = "val_256.tar"
    file_md5 = "e27b17d8d44f4af9a78502beb927f808"
    url = "http://data.csail.mit.edu/places/places365/val_256.tar"
    # size: 36500

    def __init__(
        self,
        root: str,
        split=None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        self.root = os.path.expanduser(root)
        self.dataset_folder = os.path.join(self.root, self.base_folder)
        self.archive = os.path.join(self.root, self.filename)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")
        super().__init__(root, transform=transform, target_transform=target_transform, **kwargs)

    def _check_integrity(self) -> bool:
        return check_integrity(self.archive, self.file_md5)

    def _check_exists(self) -> bool:
        return os.path.exists(self.dataset_folder)

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            return
        download_and_extract_archive(
            self.url, download_root=self.root, extract_root=self.dataset_folder, md5=self.file_md5
        )
