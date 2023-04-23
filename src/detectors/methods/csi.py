from typing import Optional

import torch
from torch import Tensor, nn

from detectors.methods.templates import DetectorWithFeatureExtraction


class CSI(DetectorWithFeatureExtraction):
    """CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances

    Extract features from the last layer of a self supervised model.

    References:
        [1] https://arxiv.org/abs/2007.08176
        [2] https://github.com/alinlab/CSI
    """

    def __init__(self, model: nn.Module, **kwargs) -> None:
        super().__init__(
            model,
            features_nodes=[list(model._modules.keys())[-1]],
            all_blocks=False,
            last_layer=False,
            pooling_op_name="none",
            aggregation_method_name="none",
        )

    def _fit_params(self) -> None:
        self.nearest_neighbors = self.train_features

    def _layer_score(self, features: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None, **kwargs):
        den = torch.norm(features, dim=-1, keepdim=True) * torch.norm(
            self.nearest_neighbors[layer_name], dim=-1, keepdim=True
        )
        num = features @ self.nearest_neighbors[layer_name].T
        stack = num / (den + 1e-7)
        return torch.norm(features, p=2, dim=-1) * torch.max(stack, dim=-1)[0]  # type: ignore
