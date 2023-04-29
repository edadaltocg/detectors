import logging
from typing import List, Optional

import faiss
import numpy as np
import torch
from torch import Tensor, nn

from detectors.methods.templates import DetectorWithFeatureExtraction

_logger = logging.getLogger(__name__)

HYPERPARAMETERS = dict(k=dict(low=1, high=100, step=5), avg_topk=[True, False])


class KnnEuclides(DetectorWithFeatureExtraction):
    """K-NN detector based on Euclidean distance.

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

    References:
        [1] https://arxiv.org/abs/2204.06507
    """

    def __init__(
        self,
        model: nn.Module,
        features_nodes: Optional[List[str]] = None,
        all_blocks: bool = False,
        last_layer: bool = False,
        pooling_op_name: str = "avg",
        aggregation_method_name="mean",
        alpha: float = 1,
        k: int = 10,
        avg_topk: bool = False,
        **kwargs,
    ):
        super().__init__(
            model,
            features_nodes=features_nodes,
            all_blocks=all_blocks,
            last_layer=last_layer,
            pooling_op_name=pooling_op_name,
            aggregation_method_name=aggregation_method_name,
        )
        self.alpha = alpha
        self.k = k
        self.mean_op = avg_topk

        assert 0 < self.alpha <= 1, "alpha must be in the interval (0, 1]"

    def _layer_score(self, x: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None):
        x = x / torch.norm(x, p=2, dim=-1, keepdim=True)  # type: ignore
        topk, _ = self.index[layer_name].search(x.cpu().numpy().astype(np.float32), k=self.k)
        topk = torch.from_numpy(topk).to(x.device)
        # pairwise = torch.cdist(x, self.ref[layer_name].to(x.device), p=2)
        # topk, _ = torch.topk(pairwise, k=self.k, dim=-1, largest=False)
        if self.mean_op:
            return -topk.mean(dim=-1)
        else:
            return -topk[:, -1]

    def _fit_params(self) -> None:
        self.ref = {}
        self.index = {}
        for layer_name, layer_features in self.train_features.items():
            self.ref[layer_name] = layer_features[
                torch.randperm(layer_features.shape[0])[: int(self.alpha * layer_features.shape[0])]
            ]
            # normalize train features
            self.ref[layer_name] = self.ref[layer_name] / torch.norm(
                self.ref[layer_name], p="fro", dim=-1, keepdim=True
            )
            self.index[layer_name] = faiss.IndexFlatL2(self.ref[layer_name].shape[1])
            self.index[layer_name].add(self.ref[layer_name].cpu().numpy().astype(np.float32))
