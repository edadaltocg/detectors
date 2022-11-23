import os
import subprocess
from typing import Callable, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, extract_archive, verify_str_arg


class iSUN(ImageFolder):
    base_folder = "iSUN"
    images_folder = ""
    filename = "iSUN.tar.gz"
    file_md5 = "be77b0f2c26fda898afac5f99645ee70"
    url = "https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz"
    splits = ("all", "test")
    # size 8925

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
        self.split = verify_str_arg(split, "split", self.splits)
        print(self.split_folder)
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


def test():
    import torch.utils.data
    import torchvision

    transforms = torchvision.transforms.ToTensor()
    dataset = iSUN("./data", download=True, transform=transforms)
    print(iSUN)
    print(dataset.dataset_folder)
    print(dataset[0])
    print(len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset)
    for img, label in data_loader:
        print(img.shape)
        print(label)
        break


if __name__ == "__main__":
    test()
