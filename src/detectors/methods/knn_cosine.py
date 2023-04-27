import logging
from typing import Optional

import torch
from torch import Tensor

from detectors.methods.knn_euclides import KnnEuclides

_logger = logging.getLogger(__name__)

HYPERPARAMETERS = dict(k=dict(low=1, high=100, step=5), avg_topk=[True, False])


class KnnCosine(KnnEuclides):
    """K-NN detector based on cosine similarity.

    Args:
        model (nn.Module): Model to be used to extract features
        features_nodes (Optional[List[str]]): List of strings that represent the feature nodes.
            Defaults to None.
        all_blocks (bool, optional): If True, use all blocks of the model. Defaults to False.
        last_layer (bool, optional): If True, use also the last layer of the model. Defaults to False.
        pooling_op_name (str, optional): Pooling operation to be applied to the features.
            Can be one of ["max", "avg", "flatten", "getitem", "none"]. Defaults to "avg".
        aggregation_method_name (str, optional): Aggregation method to be applied to the features. Defaults to None.
        alpha (float, optional): Alpha parameter for the input pre-processing. Defaults to 1.
        k (int, optional): Number of nearest neighbors to be considered. Defaults to 10.
        avg_topk (bool, optional): If True, average the top-k scores. Defaults to False.
    """

    def _layer_score(self, x: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None):
        x = x / torch.norm(x, p=2, dim=-1, keepdim=True)  # type: ignore
        pairwise = x @ self.ref[layer_name].to(x.device).T
        _logger.debug("Pairwise shape: %s", pairwise.shape)
        topk, _ = torch.topk(pairwise, k=self.k, dim=-1)
        if self.mean_op:
            return topk.mean(dim=-1)
        else:
            return topk[:, -1]
