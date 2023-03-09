import os
from typing import Callable, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

MOS_PAPER_URL = ""


class MOSSUN(ImageFolder):
    """`MOS SUN <MOS_PAPER_URL>`_ Dataset subset.

    Args:
        root (string): Root directory of dataset where directory
            exists or will be saved to if download is set to True.
        split (string, optional): The dataset split, not used.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop`.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        **kwargs: Additional arguments passed to :class:`~torchvision.datasets.ImageFolder`.
    """

    base_folder = "mos_sun"
    filename = "SUN.tar.gz"
    file_md5 = "8469c3ada62211477954ec1be53b12d0"
    url = "http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz"
    # size: 10000

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


class MOSPlaces365(MOSSUN):
    """`MOS Places365 <MOS_PAPER_URL>`_ Dataset subset.

    Args:
        root (string): Root directory of dataset where directory
            exists or will be saved to if download is set to True.
        split (string, optional): The dataset split, not used.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop`.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        **kwargs: Additional arguments passed to :class:`~torchvision.datasets.ImageFolder`.
    """

    base_folder = "mos_places365"
    filename = "Places.tar.gz"
    file_md5 = "b5cb5eba2754ae2a28beea8718db699a"
    url = "http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz"


class MOSiNaturalist(MOSSUN):
    """`MOS iNaturalist <MOS_PAPER_URL>`_ Dataset subset.

    Args:
        root (string): Root directory of dataset where directory
            exists or will be saved to if download is set to True.
        split (string, optional): The dataset split, not used.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop`.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        **kwargs: Additional arguments passed to :class:`~torchvision.datasets.ImageFolder`.
    """

    base_folder = "mos_inaturalist"
    filename = "iNaturalist.tar.gz"
    file_md5 = "5be6ea8aa027d7b631916427b32cb335"
    url = "http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz"
