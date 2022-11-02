import os
from typing import Any, Callable, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg


class EnglishChars(ImageFolder):
    """In the English language, Latin script (excluding accents) and Hindu-Arabic numerals are used.
    For simplicity we call this the "English" characters set. Our dataset consists of:

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
    splits = ("all", "test")

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
        self.split = verify_str_arg(split, "split", self.splits)

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
            self.url,
            download_root=self.root,
            extract_root=self._dataset_folder,
            remove_finished=False,
            md5=self.file_md5,
        )


def test():
    import torch.utils.data
    import torchvision

    transforms = torchvision.transforms.ToTensor()
    dataset = EnglishChars("./data", split="all", download=True, transform=transforms)
    print(EnglishChars)
    print(dataset._dataset_folder)
    print(dataset[0])
    print(len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset)
    for img, label in data_loader:
        print(img.shape)
        print(label)
        break


if __name__ == "__main__":
    test()
