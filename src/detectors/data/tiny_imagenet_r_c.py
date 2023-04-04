import os
import subprocess
from typing import Callable, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, extract_archive


class TinyImageNetResized(ImageFolder):
    base_folder = "Imagenet_resized"
    filename = "Imagenet_resize.tar.gz"
    file_md5 = "0f9ff11d45babf2eff5fe12281d1ac31"
    url = "https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz"

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
        print(f"Downloading {self.filename}...")
        subprocess.run(f"wget {self.url} -P {self.root}".split(" "), capture_output=True, text=True)
        extract_archive(self.archive, self.dataset_folder)


class TinyImageNetCroped(TinyImageNetResized):
    base_folder = "Imagenet_croped"
    filename = "Imagenet.tar.gz"
    file_md5 = "7c0827e4246c3718a5ee076e999e52e5"
    url = "https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz"
