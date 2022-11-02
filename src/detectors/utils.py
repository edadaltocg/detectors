import json
from typing import Any, Dict, Iterable, List, Tuple, Type, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset


def str_to_dict(string: str) -> Dict[str, Any]:
    string = string.strip().strip("'")
    return json.loads(string)


class ConcatDatasetsDim1(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets

        assert all(len(ds) for ds in self.datasets)

    def __getitem__(self, index):
        outputs = []
        for dataset in self.datasets:
            elem = dataset.__getitem__(index)
            if isinstance(elem, (list, tuple)):
                outputs += elem
            else:
                outputs += [elem]
        return tuple(outputs)

    def __len__(self):
        return len(self.datasets[0])
