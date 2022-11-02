import os
from typing import Any, Callable, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg


class Places365(ImageFolder):
    base_folder = "places365"
    images_folder = ""
    filename = "val_256.tar"
    file_md5 = "e27b17d8d44f4af9a78502beb927f808"
    url = "http://data.csail.mit.edu/places/places365/val_256.tar"
    splits = ("all", "test")
    # size: 36500
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
    dataset = Places365("./data", split="all", download=True, transform=transforms)
    print(Places365)
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
