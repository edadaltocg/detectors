from functools import partial

import torch
from torch import Tensor, nn

from detectors.methods.utils import input_pre_processing

HYPERPARAMETERS = dict(temperature=dict(low=1, high=1000, step=1.0))


def _score_fn(x: Tensor, model: nn.Module, temperature: float = 1.0, **kwargs) -> Tensor:
    logits = model(x)
    return temperature * torch.logsumexp(logits / temperature, dim=-1)


def energy(x: Tensor, model: nn.Module, temperature: float = 1.0, eps: float = 0.0, **kwargs):
    """Energy-based OOD detector.

    Args:
        x (Tensor): input tensor.
        model (nn.Module): classifier.
        temperature (float, optional): softmax temperature parameter. Defaults to 1.0.

    Returns:
        Tensor: OOD scores for each input.

    References:
        [1] https://arxiv.org/abs/2010.03759
    """
    model.eval()
    if eps > 0:
        x = input_pre_processing(partial(_score_fn, model=model, temperature=temperature), x, eps)

    with torch.no_grad():
        return _score_fn(x, model, temperature)
