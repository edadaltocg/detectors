import logging
from typing import List, Optional, Union

import torch
from torch import Tensor

from detectors.methods.templates import DetectorWithFeatureExtraction

_logger = logging.getLogger(__name__)


def max_cosine_sim_layer_score(x: Tensor, mus: Union[Tensor, List[Tensor]], eps=1e-7):
    if isinstance(mus, list):
        mus = torch.cat(mus, dim=0)
    den = torch.norm(x, dim=-1, keepdim=True) @ torch.norm(mus, dim=-1, keepdim=True).T
    stack = x @ mus.T
    stack = stack / (den + eps)
    return stack.max(dim=-1)[0]


class MaxCosineSimilarity(DetectorWithFeatureExtraction):
    """`MaxCosineSimilarity <PAPER_URL>` detector.

    Args:
        model (nn.Module): Model to be used to extract features
        features_nodes (Optional[List[str]]): List of strings that represent the feature nodes.
            Defaults to None.
        all_blocks (bool, optional): If True, use all blocks of the model. Defaults to False.
        last_layer (bool, optional): If True, use also the last layer of the model. Defaults to False.
        pooling_op_name (str, optional): Pooling operation to be applied to the features. Can be one of ["max", "avg", "flatten", "getitem", "none"].
            Defaults to "avg".
        aggregation_method_name (str, optional): Aggregation method to be applied to the features. Defaults to None.
        **kwargs
    """

    def __init__(
        self,
        model: torch.nn.Module,
        features_nodes: Optional[List[str]] = None,
        all_blocks: bool = False,
        pooling_op_name: str = "max",
        aggregation_method_name=None,
        **kwargs
    ):
        super().__init__(
            model,
            features_nodes=features_nodes,
            all_blocks=all_blocks,
            pooling_op_name=pooling_op_name,
            aggregation_method_name=aggregation_method_name,
            **kwargs,
        )

    def _layer_score(self, x: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None):
        return max_cosine_sim_layer_score(x, self.mus[layer_name].to(x.device))

    def _fit_params(self) -> None:
        self.mus = {}
        device = next(self.model.parameters()).device
        unique_classes = torch.unique(self.train_targets).detach().cpu().numpy().tolist()
        for layer_name, layer_features in self.train_features.items():
            for c in unique_classes:
                filt = self.train_targets == c
                if filt.sum() == 0:
                    continue
                self.mus[layer_name].append(layer_features[filt].to(device).mean(0, keepdim=True))
            self.mus = torch.cat(self.mus[layer_name], dim=0)
