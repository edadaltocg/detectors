from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from detectors.methods.templates import DetectorWithFeatureExtraction


def igeoodlogits_vec(logits, temperature, centroids, epsilon=1e-12):
    logits = torch.sqrt(F.softmax(logits / temperature, dim=1))
    centroids = torch.sqrt(F.softmax(centroids / temperature, dim=1))
    mult = logits @ centroids.T
    stack = 2 * torch.acos(torch.clamp(mult, -1 + epsilon, 1 - epsilon))
    return stack


class IgeoodLogits(DetectorWithFeatureExtraction):
    def __init__(self, model: torch.nn.Module, temperature: float = 1.0, **kwargs):
        super().__init__(
            model, all_blocks=False, last_layer=True, pooling_op_name="none", aggregation_method_name=None, **kwargs
        )
        self.temperature = temperature

    def _layer_score(self, x: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None):
        return igeoodlogits_vec(x, temperature=self.temperature, centroids=self.mus[layer_name].to(x.device)).mean(
            dim=1
        )

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
