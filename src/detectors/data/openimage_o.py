import os
from typing import Callable, Optional
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url


class OpenImageO(ImageFolder):
    """OpenImageO dataset.

    - Length: <17632 (sore url returns http error code 404)
    - Size: >28GB
    - Paper: https://arxiv.org/pdf/2203.10807.pdf
    - Auxiliary file: `OpenImageO/openimage_o_urls.csv`

    Args:
    """

    base_folder = "openimage-o"
    filenames = "OpenImageO/openimage_o_urls.csv"
    # size: 17632

    def __init__(
        self, root: str, split=None, transform: Optional[Callable] = None, download: bool = False, **kwargs
    ) -> None:
        self.root = os.path.expanduser(root)
        self.dataset_folder = os.path.join(self.root, self.base_folder)
        self.id_urls = [
            l.split(",") for l in open(os.path.join(os.path.dirname(__file__), self.filenames)).read().splitlines()[1:]
        ]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        super().__init__(self.dataset_folder, transform=transform, **kwargs)

    def _check_integrity(self) -> bool:
        # assert number of iumages in folder is equal to 17632
        if not self._check_exists():
            return False

        count = 0
        # Iterate directory
        for path in os.listdir(self.dataset_folder):
            # check if current path is a file
            if os.path.isfile(os.path.join(self.dataset_folder, path)):
                count += 1
        return count >= 15_000

    def _check_exists(self) -> bool:
        return os.path.exists(self.dataset_folder)

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            return

        def download_url_wrapper(args):
            url, id, root = args
            extension = url.split(".")[-1]
            filename = id + "." + extension
            if not os.path.exists(os.path.join(root, filename)):
                try:
                    download_url(url, root, filename=filename, max_redirect_hops=5)
                except Exception as e:
                    print(f"Error downloading: {url}. Error: {e}")

        args = [(url, id, os.path.join(self.dataset_folder, self.base_folder)) for id, url in self.id_urls]
        cpus = cpu_count()
        with ThreadPool(cpus - 1) as pool:
            results = pool.map_async(download_url_wrapper, args)
            results.wait()

            pool.close()
            pool.join()
