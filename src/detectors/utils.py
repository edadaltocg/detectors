import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd
import torch
import torch.distributed as dist
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


def sync_tensor_across_gpus(t: torch.Tensor) -> torch.Tensor:
    """Gather tensor from all gpus and return a tensor with dim 0 equal to the number of gpus.

    Args:
        t (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_

    References:
        https://discuss.pytorch.org/t/ddp-evaluation-gather-output-loss-and-stuff-how-to/130593/2
    """
    if not dist.is_initialized():
        return t
    group = dist.group.WORLD
    group_size = dist.get_world_size(group)
    if group_size == 1:
        return t
    gather_t_tensor = [torch.zeros_like(t) for _ in range(group_size)]
    dist.all_gather(gather_t_tensor, t)  # this works with nccl backend when tensors need to be on gpu.
    # for gloo and mpi backends, tensors need to be on cpu. also this works single machine with
    # multiple   gpus. for multiple nodes, you should use dist.all_gather_multigpu. both have the
    # same definition... see [here](https://pytorch.org/docs/stable/distributed.html).
    #  somewhere in the same page, it was mentioned that dist.all_gather_multigpu is more for
    # multi-nodes. still dont see the benefit of all_gather_multigpu. the provided working case in
    # the doc is  vague...
    # move tensors to cpu
    # gather_t_tensor = [t.cpu() for t in gather_t_tensor]
    gather_t_tensor = torch.cat(gather_t_tensor, dim=0)

    return gather_t_tensor
