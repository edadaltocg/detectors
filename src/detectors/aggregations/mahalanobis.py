from torch import Tensor
import torch
import logging

_logger = logging.getLogger(__name__)


def mahalanobis_distance_inv_fast(x: Tensor, y: Tensor, precision: Tensor):
    """Mahalanobis distance betwee x and y with an accelerated implementation.

    Args:
        x (Tensor): first point.
        y (Tensor): second point.
        precision (Tensor): inverse of the covariance matrix.
    """
    d_squared = torch.mm(torch.mm(x - y, precision), (x - y).T).diag()
    return torch.sqrt(d_squared)


class MahalanobisAggregation:
    def __init__(self, *args, **kwargs) -> None:
        self.mu = None
        self.pinv = None

    def fit(self, stack: Tensor, *args, **kwargs):
        self.mu = stack.mean(dim=0, keepdim=True)
        self.pinv = torch.linalg.pinv(torch.cov(stack.T))

    def __call__(self, scores: Tensor, *args, **kwargs):
        return -mahalanobis_distance_inv_fast(scores, self.mu.to(scores.device), self.pinv.to(scores.device))
