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
    """MaxCosineSimilarity detector.


    Args:
        model (nn.Module): Model to be used to extract features
        features_nodes (Optional[List[str]]): List of strings that represent the feature nodes.
            Defaults to None.
        all_blocks (bool, optional): If True, use all blocks of the model. Defaults to False.
        last_layer (bool, optional): If True, use also the last layer of the model. Defaults to False.
        pooling_op_name (str, optional): Pooling operation to be applied to the features.
            Can be one of:
                `max`, `avg`, `none`, `flatten`, `getitem`, `avg_or_getitem`, `max_or_getitem`.
            Defaults to "avg".
        aggregation_method_name (str, optional): Aggregation method to be applied to the features. Defaults to None.

    References:
        [1] https://openaccess.thecvf.com/content/ACCV2020/html/Techapanurak_Hyperparameter-Free_Out-of-Distribution_Detection_Using_Cosine_Similarity_ACCV_2020_paper.html
    """

    def _layer_score(self, x: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None):
        return max_cosine_sim_layer_score(x, self.mus[layer_name].to(x.device))

    def _fit_params(self) -> None:
        self.mus = {}
        unique_classes = torch.unique(self.train_targets).detach().cpu().numpy().tolist()
        for layer_name, layer_features in self.train_features.items():
            self.mus[layer_name] = []
            for c in unique_classes:
                filt = self.train_targets == c
                if filt.sum() == 0:
                    continue
                self.mus[layer_name].append(layer_features[filt].mean(0, keepdim=True))
            self.mus[layer_name] = torch.cat(self.mus[layer_name], dim=0)
