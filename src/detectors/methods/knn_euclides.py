import logging
from typing import List, Optional

import torch
from torch import Tensor, nn

from detectors.methods.templates import DetectorWithFeatureExtraction

_logger = logging.getLogger(__name__)


class KnnEuclides(DetectorWithFeatureExtraction):
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
        pairwise = torch.cdist(x, self.ref[layer_name].to(x.device), p=2)
        topk, _ = torch.topk(pairwise, k=self.k, dim=-1, largest=False)
        if self.mean_op:
            return -topk.mean(dim=-1)
        else:
            return -topk[:, -1]

    def _fit_params(self) -> None:
        self.ref = {}
        for layer_name, layer_features in self.train_features.items():
            self.ref[layer_name] = layer_features[
                torch.randperm(layer_features.shape[0])[: int(self.alpha * layer_features.shape[0])]
            ]
            # normalize train features
            self.ref[layer_name] = self.ref[layer_name] / torch.norm(self.ref[layer_name], p=2, dim=-1, keepdim=True)  # type: ignore
