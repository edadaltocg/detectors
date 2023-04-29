import logging
from typing import List, Optional, Union

import torch
from torch import Tensor
from tqdm import tqdm

from detectors.methods.templates import DetectorWithFeatureExtraction

_logger = logging.getLogger(__name__)


def projection_layer_score(x: Tensor, mus: Union[Tensor, List[Tensor]], eps=1e-7):
    if isinstance(mus, list):
        mus = torch.cat(mus, dim=0)
    den = torch.norm(x, dim=-1, keepdim=True) @ torch.norm(mus, dim=-1, keepdim=True).T
    stack = x @ mus.T
    stack = stack / (den + eps)
    return torch.norm(x, p=2, dim=-1, keepdim=True) * stack  # type: ignore


class Projection(DetectorWithFeatureExtraction):
    def __init__(
        self,
        model: torch.nn.Module,
        features_nodes: Optional[List[str]] = None,
        pooling_op_name: str = "max_or_getitem",
        **kwargs
    ):
        super().__init__(
            model=model,
            features_nodes=features_nodes,
            pooling_op_name=pooling_op_name,
            all_blocks=True,
            last_layer=True,
            aggregation_method_name="innerprod",
            **kwargs
        )

        self.mus = None

    def end(self, *args, **kwargs):
        for node_name, v in self.train_features.items():
            if isinstance(v, list):
                self.train_features[node_name] = torch.cat(v, dim=0)
        if isinstance(self.train_targets, list):
            self.train_targets = torch.cat(self.train_targets, dim=0)

        self._fit_params()

        if self.aggregation_method is not None and hasattr(self.aggregation_method, "fit"):
            _logger.info("Fitting aggregator %s...", self.aggregation_method_name)
            self.batch_size = self.train_targets.shape[0]  # type: ignore
            all_scores = torch.zeros(self.train_targets.shape[0], len(self.train_features))
            train_probs = torch.softmax(self.train_features[list(self.train_features.keys())[-1]], 1)
            for i, (k, v) in tqdm(enumerate(self.train_features.items())):
                idx = 0
                for idx in range(0, v.shape[0], self.batch_size):
                    all_scores[:, i] = self._layer_score(
                        v[idx : idx + self.batch_size], train_probs[idx : idx + self.batch_size], k, i
                    ).view(-1)
            self.aggregation_method.fit(all_scores, self.train_targets)

        del self.train_features
        del self.train_targets

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        features = self.feature_extractor(x)
        logits = features[list(features.keys())[-1]]
        probs = torch.softmax(logits, dim=1)
        all_scores = torch.zeros(x.shape[0], len(features), device=x.device)
        for i, (k, v) in enumerate(features.items()):
            all_scores[:, i] = self._layer_score(v, probs, k, i).view(-1)

        all_scores = self.aggregation_method(all_scores)
        return all_scores.view(-1)

    def _layer_score(self, x: Tensor, probs: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None):
        scores = projection_layer_score(x, self.mus[layer_name].to(x.device))  # type: ignore
        scores = torch.sum(scores * probs, 1, keepdim=True)
        return scores

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
