import os
from typing import Callable, Dict, List, Optional, Tuple

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

PAPER_URL = "https://arxiv.org/pdf/2306.00826.pdf"


class NINCOFull(ImageFolder):
    """`NINCO` Dataset subset.

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

    base_folder = "ninco"
    filename = "NINCO_all.tar.gz"
    file_md5 = "b9ffae324363cd900a81ce3c367cd834"
    url = "https://zenodo.org/record/8013288/files/NINCO_all.tar.gz"
    # size: 15393

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
        return check_integrity(self.archive, self.file_md5)

    def _check_exists(self) -> bool:
        return os.path.exists(self.dataset_folder)

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            return
        download_and_extract_archive(
            self.url, download_root=self.root, extract_root=self.dataset_folder, md5=self.file_md5
        )


class NINCO(NINCOFull):
    base_folder = "ninco/NINCO/NINCO_OOD_classes/"

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            return
        download_and_extract_archive(
            self.url, download_root=self.root, extract_root=os.path.join(self.root, "ninco"), md5=self.file_md5
        )


class SSBHard(NINCO):
    base_folder = "ninco/NINCO/NINCO_popular_datasets_subsamples/SSB_hard"

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [""]

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class SSBEasy(SSBHard):
    base_folder = "ninco/NINCO/NINCO_popular_datasets_subsamples/SSB_hard"


class TexturesClean(SSBHard):
    base_folder = "ninco/NINCO/NINCO_popular_datasets_subsamples/Textures"


# ninco/NINCO/NINCO_popular_datasets_subsamples/
