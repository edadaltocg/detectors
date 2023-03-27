from torch import Tensor
import torch
import logging

_logger = logging.getLogger(__name__)


class CosineAggregation:
    def __init__(self, *args, **kwargs) -> None:
        self.mu = None

    def fit(self, stack: Tensor, *args, **kwargs):
        self.mu = stack.mean(dim=0, keepdim=True)

    def __call__(self, scores: Tensor, *args, **kwargs):
        return torch.nn.functional.cosine_similarity(scores, self.mu.to(scores.device), dim=1)
