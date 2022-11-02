from functools import partial

import torch
import torch.backends.mps
from detectors.methods.ood import msp
from detectors.pipelines.ood import OODCifar10Pipeline


def test_pipeline_ood_cifar10():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    batch_size = 64
    pipeline = OODCifar10Pipeline(device)
    model = ...
    msp_method = partial(msp, model=model)

    pipeline = pipeline("ood-cifar10", device, batch_size=batch_size)
    pipeline.benchmark({"msp": msp_method})
    return


if __name__ == "__main__":
    test_pipeline_ood_cifar10()
