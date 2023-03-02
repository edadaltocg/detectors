import os
from typing import Callable, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class Textures(ImageFolder):
    """DTD is a texture database, consisting of 5640 images, organized according
    to a list of 47 terms (categories) inspired from human perception. There are 120 images
    for each category. Image sizes range between 300x300 and 640x640, and the images
    contain at least 90% of the surface representing the category attribute. The images
    were collected from Google and Flickr by entering our proposed attributes and related
    terms as search queries."""

    base_folder = "textures"
    images_folder = "dtd/images"
    filename = "dtd-r1.0.1.tar.gz"
    file_md5 = "fff73e5086ae6bdbea199a49dfb8a4c1"
    url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"

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

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")

        super().__init__(self.split_folder, transform=transform, target_transform=target_transform, **kwargs)

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
        download_and_extract_archive(
            self.url, download_root=self.root, extract_root=self.dataset_folder, md5=self.file_md5
        )
