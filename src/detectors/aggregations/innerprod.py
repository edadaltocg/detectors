import logging

import torch
from torch import Tensor

_logger = logging.getLogger(__name__)


class InnerProductAggregation:
    def __init__(self, *args, **kwargs) -> None:
        self.max_trajectory = None
        self.ref_trajectory = None
        self.scale = None

    def fit(self, stack: Tensor, *args, **kwargs):
        self.max_trajectory = stack.max(dim=0, keepdim=True)[0]
        self.ref_trajectory = stack.mean(dim=0, keepdim=True) / self.max_trajectory
        self.scale = torch.sum(self.ref_trajectory ** 2)

        _logger.debug("InnerProductAggregation parameters")
        _logger.debug(f"max_trajectory: {self.max_trajectory}")
        _logger.debug(f"ref_trajectory: {self.ref_trajectory}")
        _logger.debug(f"scale: {self.scale}")

    def __call__(self, scores: Tensor, *args, **kwargs):
        scores = scores / self.max_trajectory.to(scores.device)
        scores = torch.sum(scores * self.ref_trajectory.to(scores.device), dim=1) / self.scale.to(scores.device)
        return scores
