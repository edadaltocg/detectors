import logging
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from detectors.methods.utils import input_pre_processing

_logger = logging.getLogger(__name__)


HYPERPARAMETERS = dict(temperature=dict(low=0.1, high=1000, step=0.1), eps=dict(low=0.0, high=0.005, step=0.0001))


def igeoodlogits_vec(logits, temperature, centroids, epsilon=1e-12):
    logits = torch.sqrt(F.softmax(logits / temperature, dim=1))
    centroids = torch.sqrt(F.softmax(centroids / temperature, dim=1))
    mult = logits @ centroids.T
    stack = 2 * torch.acos(torch.clamp(mult, -1 + epsilon, 1 - epsilon))
    return stack


def _score_fn(x: Tensor, model: torch.nn.Module, centroids: Tensor, temperature: float = 1.0, **kwargs) -> Tensor:
    logits = model(x)
    return igeoodlogits_vec(logits, temperature, centroids).mean(dim=1)


class IgeoodLogits:
    """IGEOOD detector.

    Args:
        model (nn.Module): classifier.
        temperature (float, optional): softmax temperature parameter. Defaults to 1.0.
        eps (float, optional): input preprocessing noise value. Defaults to 0.0 (no input preprocessing).

    References:
        [1] https://arxiv.org/abs/2203.07798
    """

    def __init__(self, model: torch.nn.Module, temperature: float = 1.0, eps: float = 0.0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.eps = eps

        self.model.eval()

    @torch.no_grad()
    def start(self, example: Optional[Tensor] = None, fit_length: Optional[int] = None, *args, **kwargs):
        self.train_features = []
        self.train_targets = []
        self.mus = []
        self.idx = 0
        if example is not None and fit_length is not None:
            logits = self.model(example)
            self.train_features = torch.zeros((fit_length,) + logits.shape[1:], dtype=logits.dtype)
            self.train_targets = torch.ones((fit_length,), dtype=torch.long) * -1

    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor, *args, **kwargs):
        self.batch_size = x.shape[0]
        logits = self.model(x)

        if isinstance(self.train_features, list):
            self.train_features.append(logits)
        else:
            self.train_features[self.idx : self.idx + logits.shape[0]] = logits

        if isinstance(self.train_targets, list):
            self.train_targets.append(y)
        else:
            self.train_targets[self.idx : self.idx + y.shape[0]] = y

        self.idx += y.shape[0]

    def end(self, *args, **kwargs):
        if isinstance(self.train_features, list):
            self.train_features = torch.cat(self.train_features, dim=0)
        else:
            self.train_features = self.train_features[: self.idx]
        if isinstance(self.train_targets, list):
            self.train_targets = torch.cat(self.train_targets, dim=0)
        else:
            self.train_targets = self.train_targets[: self.idx]
        assert torch.all(self.train_targets > -1), "Not all targets were updated"

        self._fit_params()

        del self.train_features
        del self.train_targets

    def _fit_params(self) -> None:
        self.mus = []
        unique_classes = torch.unique(self.train_targets).detach().cpu().numpy().tolist()
        for c in unique_classes:
            filt = self.train_targets == c
            if filt.sum() == 0:
                continue
            self.mus.append(self.train_features[filt].mean(0, keepdim=True))
        self.mus = torch.cat(self.mus, dim=0)

    def __call__(self, x: Tensor) -> Tensor:
        self.mus = self.mus.to(x.device)
        if self.eps > 0:
            x = input_pre_processing(
                partial(_score_fn, model=self.model, temperature=self.temperature, centroids=self.mus), x, self.eps
            )

        with torch.no_grad():
            return _score_fn(x, self.model, self.mus, temperature=self.temperature)
