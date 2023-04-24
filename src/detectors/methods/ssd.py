from typing import Optional

import faiss
import torch
from torch import Tensor, nn

from detectors.methods.mahalanobis import mahalanobis_distance_inv
from detectors.methods.templates import DetectorWithFeatureExtraction


HYPERPARAMETERS = dict(nclusters=dict(low=1, high=2048, step=1))


class SSD(DetectorWithFeatureExtraction):
    """SSD: A Unified Framework for Self-Supervised Outlier Detection.

    Extract features from the last layer of a self supervised model.

    References:
        [1] https://arxiv.org/abs/2103.12051
        [2] https://github.com/inspire-group/SSD
    """

    def __init__(
        self, model: nn.Module, nclusters: Optional[int] = None, niter: int = 300, cov_reg=1e-8, **kwargs
    ) -> None:
        super().__init__(
            model,
            features_nodes=[list(model._modules.keys())[-1]],
            all_blocks=False,
            last_layer=False,
            pooling_op_name="none",
            aggregation_method_name="none",
        )
        self.nclusters = nclusters
        self.niter = niter
        self.cov_reg = cov_reg

        self.kmeans = {}

    @torch.no_grad()
    def _fit_params(self) -> None:
        if self.nclusters is None:
            self.nclusters = torch.unique(self.train_targets).shape[0]

        self.kmeans = {}
        self.precisions = {}
        self.mus = {}
        self.global_mu = {}
        self.global_std = {}
        for k, v in self.train_features.items():
            # pre-processing
            v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-10)
            self.global_mu[k] = torch.mean(v, dim=0, keepdim=True)
            self.global_std[k] = torch.std(v, dim=0, keepdim=True)
            v = (v - self.global_mu[k]) / (self.global_std[k] + 1e-10)

            self.kmeans[k] = faiss.Kmeans(v.shape[1], self.nclusters, niter=self.niter, verbose=False, gpu=False)
            self.kmeans[k].train(v.cpu().numpy())

            # compute covariances
            _, ypred = self.kmeans[k].assign(v.cpu().numpy())
            self.precisions[k] = []
            self.mus[k] = []
            for i in range(self.nclusters):
                cov = torch.cov(v[ypred == i].T)
                self.precisions[k].append(torch.pinverse(cov + self.cov_reg * torch.eye(cov.shape[1], device=v.device)))
                self.mus[k].append(torch.mean(v[ypred == i], dim=0))

        del self.kmeans

    @torch.no_grad()
    def _layer_score(self, features: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None, **kwargs):
        features = features / (torch.norm(features, dim=-1, keepdim=True) + 1e-10)
        features = (features - self.global_mu[layer_name].to(features.device)) / (
            self.global_std[layer_name].to(features.device) + 1e-10
        )
        scores = torch.zeros((features.shape[0], len(self.mus[layer_name])), device=features.device)
        for i, (mu, pinv) in enumerate(zip(self.mus[layer_name], self.precisions[layer_name])):
            scores[:, i] = mahalanobis_distance_inv(features, mu.to(features.device), pinv.to(features.device))
        return -torch.nan_to_num(scores.min(dim=1)[0], 1e6)
