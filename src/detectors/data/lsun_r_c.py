import os
import subprocess
from typing import Callable, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, extract_archive


class LSUNResized(ImageFolder):
    """`LSUN (r) <ODIN_PAPER_URL>`_ Dataset subset.

    Args:
        root (string): Root directory of dataset where directory
            exists or will be saved to if download is set to True.
        split (string, optional): The dataset split, not used.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, `transforms.RandomCrop`.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        **kwargs: Additional arguments passed to :class:`~torchvision.datasets.ImageFolder`.
    """

    base_folder = "LSUN_resized"
    filename = "LSUN_resize.tar.gz"
    file_md5 = "278b7b31c8cb7e804a1465a8ce03a2dc"
    url = "https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz"

    def __init__(
        self, root: str, split=None, transform: Optional[Callable] = None, download: bool = False, **kwargs
    ) -> None:
        self.root = os.path.expanduser(root)
        self.dataset_folder = os.path.join(self.root, self.base_folder)
        self.archive = os.path.join(self.root, self.filename)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")
        super().__init__(self.dataset_folder, transform=transform, **kwargs)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.file_md5
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def _check_exists(self) -> bool:
        return os.path.exists(self.dataset_folder)

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            return
        print(f"Downloading {self.filename}...")

        subprocess.run(f"wget {self.url} -P {self.root}".split(" "), capture_output=True, text=True)
        extract_archive(self.archive, self.dataset_folder)


class LSUNCroped(LSUNResized):
    """`LSUN (c) <ODIN_PAPER_URL>`_ Dataset subset.

    Args:
        root (string): Root directory of dataset where directory
            exists or will be saved to if download is set to True.
        split (string, optional): The dataset split, not used.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, `transforms.RandomCrop`.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        **kwargs: Additional arguments passed to :class:`~torchvision.datasets.ImageFolder`.
    """

    base_folder = "LSUN_croped"
    filename = "LSUN.tar.gz"
    file_md5 = "458a0a0ab8e5f1cb4516d7400568e460"
    url = "https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz"
