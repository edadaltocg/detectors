import itertools
import json
import logging
import multiprocessing as mp
import os
import time
from functools import wraps
from typing import Any, Dict, List

import pandas as pd
from torch.utils.data import Dataset

_logger = logging.getLogger(__name__)


def str_to_dict(string: str) -> Dict[str, Any]:
    string = string.strip().strip("'").replace("'", '"')
    return json.loads(string)


def append_results_to_csv_file(results, filename="results.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    results = {k: [v] for k, v in results.items()}
    results = pd.DataFrame.from_dict(results, orient="columns")

    if not os.path.isfile(filename):
        results.to_csv(filename, header=True, index=False)
    else:  # it exists, so append without writing the header
        results.to_csv(filename, mode="a", header=False, index=False)


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
        start = time.perf_counter()
        outputs = func(*args, **kwargs)
        end = time.perf_counter()
        _logger.info("Function %s took %.4f s" % (func.__name__, end - start))
        return outputs

    return _time_it


def run_parallel(input_space, wrapper_fn):
    p = mp.Pool()
    input = itertools.product(*input_space)
    _logger.info(f"Input space of size {len(list(itertools.product(*input_space)))}")

    results = p.map(wrapper_fn, input)
    p.close()
    p.join()
    return results
