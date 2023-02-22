import itertools
import json
import logging
import multiprocessing as mp
from functools import wraps
from time import time
from typing import Any, Dict, List

from torch.utils.data import Dataset

_logger = logging.getLogger(__name__)


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


def timeit(func):
    # time in seconds
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        end = time()
        return end - start

    return _time_it


def run_parallel(input_space, wrapper_fn):
    p = mp.Pool()
    input = itertools.product(*input_space)
    _logger.info(f"Input space of size {len(list(itertools.product(*input_space)))}")

    results = p.map(wrapper_fn, input)
    p.close()
    p.join()
    return results
