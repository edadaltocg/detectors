from typing import Optional

import faiss
import torch
from torch import Tensor, nn

from detectors.methods.mahalanobis import mahalanobis_distance_inv
from detectors.methods.templates import DetectorWithFeatureExtraction


HYPERPARAMETERS = dict(nclusters=dict(low=1, high=2048, step=1), niter=[100, 200, 300])


class SSD(DetectorWithFeatureExtraction):
    """SSD: A Unified Framework for Self-Supervised Outlier Detection.

    Extract features from the last layer of a self supervised model.

    References:
        [1] https://arxiv.org/abs/2103.12051
        [2] https://github.com/inspire-group/SSD
    """

    def __init__(
        self, model: nn.Module, nclusters: Optional[int] = None, niter: int = 100, cov_reg=1e-6, **kwargs
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

    def _fit_params(self) -> None:
        if self.nclusters is None:
            self.nclusters = torch.unique(self.train_targets).shape[0]

        self.kmeans = {}
        for k, v in self.train_features.items():
            self.kmeans[k] = faiss.Kmeans(v.shape[1], self.nclusters, niter=self.niter, verbose=False, gpu=False)
            self.kmeans[k].train(v.numpy())

            # compute covariances
            _, ypred = self.kmeans[k].assign(v.numpy())
            self.precisions = []
            for i in range(self.nclusters):
                cov = torch.cov(v[ypred == i].T)
                self.precisions.append(torch.pinverse(cov + self.cov_reg * torch.eye(cov.shape[1], device=cov.device)))

    def _layer_score(self, features: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None, **kwargs):
        assert self.kmeans is not None, "Kmeans not initialized, you should call fit first."
        scores = torch.zeros((features.shape[0], len(self.kmeans[layer_name].centroids)), device=features.device)
        for i, (mu, pinv) in enumerate(zip(self.kmeans[layer_name].centroids, self.precisions)):
            scores[:, i] = mahalanobis_distance_inv(features, mu, pinv)
        return -torch.nan_to_num(scores.min(dim=1)[0], nan=1e6)
