import logging
import os
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def image_dataset_to_npz(DatasetCls, root: str, split: str, **kwargs) -> None:
    dataset = DatasetCls(root, split, download=True, **kwargs)
    dest_folder = DatasetCls.base_folder + "_npz"
    image_example = dataset[0][0]
    width, height = image_example.size
    _logger.info("Image size: %d x %d", width, height)
    x = np.ndarray(shape=(len(dataset), height, width, 3), dtype=np.uint8)
    y = np.ndarray(shape=(len(dataset)), dtype=np.int32)
    for i in tqdm(range(len(dataset))):
        image, label = dataset[i]
        x[i] = image
        y[i] = label

    os.makedirs(os.path.join(root, dest_folder), exist_ok=True)
    np.savez(os.path.join(root, dest_folder, f"{split}.npz"), x=x, y=y)


class DatasetNpz(Dataset):
    def __init__(
        self,
        root: str,
        base_folder_name: str,
        split: str,
        transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = os.path.expanduser(root)
        self.base_folder_name = base_folder_name
        self.split = split
        self.path = os.path.join(self.root, self.base_folder_name, f"{split}.npz")
        self.transform = transform

        data = np.load(self.path, mmap_mode="r")
        self.images = data["x"]
        self.labels = data["y"]

    def __getitem__(self, index):
        x = self.images[index]
        x = Image.fromarray(x)

        if self.transform:
            x = self.transform(x)

        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.images)

    def _check_exists(self) -> bool:
        return os.path.exists(self.path)
