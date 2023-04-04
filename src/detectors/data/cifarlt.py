"""
Adapted from https://github.com/Megvii-Nanjing/BBN
"""

from typing import Callable, Literal, Optional

import numpy as np
import torchvision
from PIL import Image

# CIFAR10:
# many: 0,1,2
# median: 3,4,5,6
# few: 7,8,9


class CIFAR10LT(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(
        self,
        root: str,
        train: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download=True,
        imbalance_ratio=0.01,
        imb_type: Literal["exp", "step"] = "exp",
    ):
        super().__init__(root, train, transform=transform, target_transform=target_transform, download=download)
        self.train = train
        if self.train:
            img_num_list = self._get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            self._gen_imbalanced_data(img_num_list)

        self.labels = self.targets

    def _get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == "exp":
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        self.img_num_per_cls = img_num_per_cls
        return img_num_per_cls

    def _gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            # np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.labels)


class CIFAR100LT(CIFAR10LT):
    cls_num = 100
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]
    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


def test():
    cifar10 = torchvision.datasets.CIFAR10(root="data", train=True, download=True)
    cifar10lt = CIFAR10LT(root="data", train=True, download=True, imb_type="exp", imbalance_ratio=0.01)
    cifar100 = torchvision.datasets.CIFAR100(root="data", train=True, download=True)
    cifar100lt = CIFAR100LT(root="data", train=True, download=True, imb_type="exp", imbalance_ratio=0.01)

    print(len(cifar10))

    print(len(cifar10lt))
    print(cifar10lt.img_num_per_cls)

    print(len(cifar100))

    print(len(cifar100lt))
    print(cifar100lt.img_num_per_cls)


if __name__ == "__main__":
    test()
