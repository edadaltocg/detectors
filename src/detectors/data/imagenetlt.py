"""From https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch/blob/master/classification/data/dataloader.py"""
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

MISC_FILES_PATH = "ImageNetLT"
FILE_PATH = os.path.dirname(__file__)


class LT_Dataset(Dataset):
    def __init__(self, root: str, split: str, class_probs, transform=None):
        print(root)
        self.root = root
        self.img_path = []
        self.labels = []
        self.transform = transform
        txt_path = os.path.join(FILE_PATH, MISC_FILES_PATH, split + ".txt")
        with open(txt_path) as f:
            for line in f:
                _label = int(line.split()[1])
                self.labels.append(_label)
                self.img_path.append(os.path.join(root, line.split()[0]))
        self.labels = np.array(self.labels)
        self.img_path = np.array(self.img_path)
        num_classes = len(class_probs)
        self.img_num_per_cls = np.zeros(num_classes, dtype=np.int32)
        self.label_imgidx_dict = {}
        for i in range(num_classes):
            self.img_num_per_cls[i] = int(np.sum(self.labels == i) * class_probs[i])
            # labels that belong to class i
            labels_filt = np.where(self.labels == i)[0]
            self.label_imgidx_dict[i] = self.img_path[labels_filt][: self.img_num_per_cls[i]]

        self.paths = []
        self.targets = []
        for k, v in self.label_imgidx_dict.items():
            self.paths.extend(v)
            self.targets.extend([k] * len(v))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.targets[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


class ImageNet1kLT(LT_Dataset):
    def __init__(self, root: str, split: str, transform=None, download: bool = False, *args, **kwargs):
        n_classes = 1000
        x = np.random.exponential(scale=1.0, size=n_classes)
        self.propability_to_sample_from_class = sorted((x - min(x)) / (max(x) - min(x)), reverse=True)
        super().__init__(root, split, self.propability_to_sample_from_class, transform, *args, **kwargs)
