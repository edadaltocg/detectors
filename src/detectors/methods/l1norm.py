from typing import List, Literal, Optional

import torch
from torch import Tensor, nn

from detectors.methods.templates import DetectorWithFeatureExtraction

HYPERPARAMETERS = dict()


class L1Norm(DetectorWithFeatureExtraction):
    """L1 norm OOD detector.

    Args:
        model (nn.Module): Model to be used to extract features.
        features_nodes (Optional[List[str]]): List of strings that represent the feature nodes.
            Defaults to None.
        all_blocks (bool, optional): If True, use all blocks of the model. Defaults to False.
        last_layer (bool, optional): If True, use also the last layer of the model. Defaults to False.
        pooling_op_name (str, optional): Pooling operation to be applied to the features.
            Can be one of `max`, `avg`, `flatten`, `getitem`, `avg_or_getitem`, `max_or_getitem`, `none`. Defaults to `avg`.
        aggregation_method_name (str, optional): Aggregation method to be applied to the features. Defaults to None.

    Reference:
        - https://arxiv.org/abs/2110.00218
    """

    def __init__(
        self,
        model: nn.Module,
        features_nodes: Optional[List[str]] = None,
        all_blocks: bool = False,
        last_layer: bool = False,
        pooling_op_name: str = "avg_or_getitem",
        aggregation_method_name: Optional[str] = "mean",
        **kwargs,
    ):
        super().__init__(
            model, features_nodes, all_blocks, last_layer, pooling_op_name, aggregation_method_name, **kwargs
        )

    @torch.no_grad()
    def _layer_score(self, x: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None):
        return -x.norm(1, dim=1)
