from typing import Optional

import faiss
import torch
from torch import Tensor, nn

from detectors.methods.mahalanobis import mahalanobis_distance_inv
from detectors.methods.templates import DetectorWithFeatureExtraction


class SSD(DetectorWithFeatureExtraction):
    def __init__(self, model: nn.Module, nclusters: int = 10, niter: int = 100, **kwargs) -> None:
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
        self.kmeans = None

    def _fit_params(self) -> None:
        self.train_features = self.train_features[list(self.train_features.keys())[0]]
        self.kmeans = faiss.Kmeans(
            self.train_features.shape[1], self.nclusters, niter=self.niter, verbose=False, gpu=False
        )
        self.kmeans.train(self.train_features)

        # compute covariances
        _, ypred = self.kmeans.assign(self.train_features)
        self.precisions = []
        for i in range(self.nclusters):
            self.precisions.append(torch.pinverse(torch.cov(self.train_features[ypred == i].T)))

    def _layer_score(self, features: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None, **kwargs):
        assert self.kmeans is not None, "Kmeans not initialized, you should call fit first."
        scores = torch.zeros((features.shape[0], len(self.kmeans.centroids)), device=features.device)
        for i, (mu, pinv) in enumerate(zip(self.kmeans.centroids, self.precisions)):
            scores[:, i] = mahalanobis_distance_inv(features, mu, pinv)
        return scores.min(dim=1)[0]
