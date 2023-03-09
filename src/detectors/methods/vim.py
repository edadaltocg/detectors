import logging
from functools import partial
from typing import List, Literal, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor


class ViM:
    """Virtual Logit Matching (ViM)

    - Title: ViM: Out-Of-Distribution with Virtual-logit Matching.
    - Paper: [https://arxiv.org/abs/2203.10807](https://arxiv.org/abs/2203.10807)
    - GitHub: [https://github.com/haoqiwang/vim/](https://github.com/haoqiwang/vim/)
    """

    def __init__(
        self,
        model: nn.Module,
        d: int,
        w: torch.Tensor,
        b: torch.Tensor,
    ):
        """
        :param model: neural network to use, is assumed to output features
        :param d: dimensionality of the principal subspace
        :param w: weights :math:`W` of the last layer of the network
        :param b: biases :math:`b` of the last layer of the network
        """
        super(ViM, self).__init__()
        self.model = model
        self.n_dim = d
        self.w = w
        self.b = b
        self.u = -torch.matmul(torch.linalg.pinv(self.w), self.b)  # new origin
        self.principal_subspace = None
        self.alpha: float = None  #: the computed :math:`\alpha` value

    def _get_logits(self, features):
        return torch.matmul(features, self.w.T) + self.b

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.model(x).cpu()

        logits = self._get_logits(features)

        x_p_t = torch.norm(torch.matmul(features - self.u, self.principal_subspace), dim=-1)
        vlogit = x_p_t * self.alpha
        energy = torch.logsumexp(logits, dim=-1)
        score = -vlogit + energy

        return -torch.Tensor(score)

    def fit(self, data_loader, device="cpu"):
        """
        Extracts features and logits, computes principle subspace and alpha. Ignores OOD samples.
        """
        # extract features
        with torch.no_grad():
            features_l = []

            for x, y in data_loader:
                features = self.model(x[known].to(device)).cpu()
                features = features.view(known.sum(), -1)  # flatten
                features_l.append(features)

        features = torch.cat(features_l).numpy()
        logits = self._get_logits(features)

        try:
            from sklearn.covariance import EmpiricalCovariance
        except ImportError:
            raise Exception("You need to install sklearn to use ViM.")

        # calculate eigenvectors of the covariance matrix
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(features - self.u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)

        # select largest eigenvectors to get the principal subspace
        largest_eigvals_idx = np.argsort(eig_vals * -1)[self.n_dim :]
        self.principal_subspace = np.ascontiguousarray((eigen_vectors.T[largest_eigvals_idx]).T)

        # calculate residual
        x_p_t = np.matmul(features - self.u, self.principal_subspace)
        vlogits = torch.norm(x_p_t, dim=-1)
        self.alpha = logits.max(dim=-1).mean() / vlogits.mean()
        return self
