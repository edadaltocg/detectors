import os
from typing import Callable, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class EnglishChars(ImageFolder):
    """In the English language, Latin script (excluding accents) and Hindu-Arabic numerals are used.
    For simplicity we call this the "English" characters set. The dataset consists of:

    * 64 classes (0-9, A-Z, a-z)
    * 7705 characters obtained from natural images
    * 3410 hand drawn characters using a tablet PC
    * 62992 synthesised characters from computer fonts
    * This gives a total of over 74K images (which explains the name of the dataset).

    """

    base_folder = "chars74k"
    images_folder = "English/Img/GoodImg/Bmp/"
    filename = "EnglishImg.tgz"
    file_md5 = "85d157e0c58f998e1cda8def62bcda0d"
    url = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz"

    def __init__(
        self, root: str, split=None, transform: Optional[Callable] = None, download: bool = False, **kwargs
    ) -> None:
        self.root = os.path.expanduser(root)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")
        super().__init__(self._split_folder, transform=transform, **kwargs)

    @property
    def _dataset_folder(self):
        return os.path.join(self.root, self.base_folder)

    @property
    def _split_folder(self):
        return os.path.join(self._dataset_folder, self.images_folder)

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
            self.url, download_root=self.root, extract_root=self._dataset_folder, md5=self.file_md5
        )
