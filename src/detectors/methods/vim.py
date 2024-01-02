import logging
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from sklearn.covariance import EmpiricalCovariance
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor

from detectors.methods.utils import get_composed_attr
from detectors.utils import sync_tensor_across_gpus

_logger = logging.getLogger(__name__)


class ViM:
    """Virtual Logit Matching (ViM) detector.

    Args:
        model (torch.nn.Module): Model to be used to extract features
        last_layer_name (Optional[str]): Name of the last layer. Defaults to None.
        penultimate_layer_name (Optional[str]): Name of the penultimate layer. Defaults to None.

    References:
        [1] https://arxiv.org/abs/2203.10807
    """

    def __init__(
        self,
        model: nn.Module,
        last_layer_name: Optional[str] = None,
        penultimate_layer_name: Optional[str] = None,
        **kwargs
    ):
        self.model = model
        self.model.eval()
        self.last_layer_name = last_layer_name
        self.penultimate_layer_name = penultimate_layer_name

        # create feature extractor
        if self.penultimate_layer_name is None:
            self.penultimate_layer_name = list(self.model._modules.keys())[-2]
        self.feature_extractor = create_feature_extractor(self.model, [self.penultimate_layer_name])
        _logger.info("Penultimate layer name: %s", self.penultimate_layer_name)

        # get the model weights of the last layer
        if last_layer_name is None:
            if hasattr(self.model, "default_cfg"):
                last_layer_name = self.model.default_cfg["classifier"]
            else:
                last_layer_name = list(model._modules.keys())[-1]
        _logger.info("Last layer name: %s", last_layer_name)
        # last_layer = model._modules[last_layer_name]
        last_layer = get_composed_attr(model, last_layer_name.split("."))

        self.w = last_layer.weight.data.squeeze().clone()
        self.b = last_layer.bias.data.squeeze().clone()

        _logger.debug("w shape: %s", self.w.shape)
        _logger.debug("b shape: %s", self.b.shape)

        self.head = list(model._modules.values())[-1]

        # new origin
        self.u = -torch.matmul(torch.linalg.pinv(self.w), self.b).float()
        _logger.debug("New origin shape: %s", self.u.shape)

        self.principal_subspace = None
        self.train_features = []
        self.train_logits = []
        self.alpha = None
        self.top_k = None

    def _get_logits(self, features: Tensor) -> Tensor:
        logits = self.head(features)
        return logits

    def start(self, *args, **kwargs):
        self.principal_subspace = None
        self.train_features = None
        self.train_logits = None
        self.alpha = None
        self.top_k = None

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs):
        features = self.feature_extractor(x)[self.penultimate_layer_name]
        features = sync_tensor_across_gpus(features)
        if dist.is_initialized():
            dist.gather(features, dst=0)

        if self.train_features is None:
            self.train_features = torch.flatten(features, start_dim=1).cpu()
        else:
            self.train_features = torch.cat([self.train_features, torch.flatten(features, start_dim=1).cpu()], dim=0)

        if self.train_logits is None:
            self.train_logits = self._get_logits(features).cpu()
        else:
            self.train_logits = torch.cat([self.train_logits, self._get_logits(features).cpu()])

    def end(self):
        self.top_k = 1000 if self.train_features.shape[1] > 1500 else 512

        _logger.info("Train features shape: %s", self.train_features.shape)
        _logger.info("Train logits shape: %s", self.train_logits.shape)

        # calculate eigenvectors of the covariance matrix
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(self.train_features.cpu().numpy() - self.u.detach().cpu().numpy())
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        determinant = np.linalg.det(ec.covariance_)
        _logger.debug("Determinant: %s", determinant)
        _logger.debug("Eigen values: %s", eig_vals)

        # select largest eigenvectors to get the principal subspace
        largest_eigvals_idx = np.argsort(eig_vals * -1)[self.top_k :]
        self.principal_subspace = torch.from_numpy(
            np.ascontiguousarray((eigen_vectors.T[largest_eigvals_idx]).T)
        ).float()
        _logger.debug("Principal subspace: %s", self.principal_subspace)

        # calculate residual
        x_p_t = torch.matmul(self.train_features.cpu() - self.u.cpu(), self.principal_subspace.cpu())
        vlogits = torch.norm(x_p_t, dim=-1)
        self.alpha = self.train_logits.max(dim=-1)[0].mean() / vlogits.mean()
        _logger.debug("Alpha: %s", self.alpha)

        del self.train_features

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.feature_extractor = self.feature_extractor.to(x.device)
        features = self.feature_extractor(x)[self.penultimate_layer_name]

        logits = self._get_logits(features)

        x_p_t = torch.norm(
            torch.matmul(torch.flatten(features, 1) - self.u.to(x.device), self.principal_subspace.to(x.device)), dim=-1
        )
        vlogit = x_p_t * self.alpha
        energy = torch.logsumexp(logits, dim=-1)
        score = -vlogit + energy
        return torch.nan_to_num(score, 1e6)
