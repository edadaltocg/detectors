from functools import partial

import torch
from torch import Tensor, nn

from .utils import input_pre_processing

HYPERPARAMETERS = dict(
    temperature={"low": 0.1, "high": 1000, "step": 0.1}, eps={"low": 0.0, "high": 0.005, "step": 0.0001}
)


def _score_fn(x: Tensor, model: nn.Module, temperature: float = 1000, **kwargs) -> Tensor:
    outputs = model(x)
    return -(1 - torch.softmax(outputs / temperature, dim=1).square().sum(dim=1))



def doctor(x: Tensor, model: nn.Module, temperature: float = 1, eps: float = 0.0, **kwargs) -> Tensor:
    """Doctor detector.

    Args:
        x (Tensor): input tensor.
        model (nn.Module): classifier.
        temperature (float, optional): softmax temperature parameter. Defaults to 1000.
        eps (float, optional): input preprocessing noise value. Defaults to 0.0 (no input preprocessing).

    Returns:
        Tensor: scores for each input.

    References:
        [1] https://arxiv.org/abs/2106.02395
    """
    model.eval()

    if eps > 0:
        x = input_pre_processing(partial(_score_fn, model=model, temperature=temperature), x, eps)

    with torch.no_grad():
        return _score_fn(x, model, temperature)
