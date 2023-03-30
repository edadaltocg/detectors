import logging
from collections import defaultdict
from typing import List, Optional, Union

import torch
from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from detectors.aggregations import create_aggregation

from detectors.methods.utils import create_reduction

_logger = logging.getLogger(__name__)


def max_cosine_sim_layer_score(x: Tensor, mus: Union[Tensor, List[Tensor]], eps=1e-7):
    if isinstance(mus, list):
        mus = torch.cat(mus, dim=0)
    den = torch.norm(x, dim=-1, keepdim=True) @ torch.norm(mus, dim=-1, keepdim=True).T
    stack = x @ mus.T
    stack = stack / (den + eps)
    return stack.max(dim=-1)[0]


class MaxCosineSimilarity:
    def __init__(
        self,
        model: torch.nn.Module,
        features_nodes: Optional[List[str]] = None,
        all_blocks: bool = False,
        pooling_op_name: str = "max",
        aggregation_method_name=None,
        **kwargs
    ):
        self.model = model
        self.model.eval()
        self.pooling_op_name = pooling_op_name
        self.device = next(self.model.parameters()).device
        self.features_nodes = features_nodes
        if hasattr(self.model, "feature_info") and self.features_nodes is None and all_blocks:
            self.features_nodes = [fi["module"] for fi in self.model.feature_info][1:]
        if self.features_nodes is None:
            self.features_nodes = [list(self.model._modules.keys())[-2]]
        _logger.info("Using features nodes: %s", self.features_nodes)
        self.feature_extractor = create_feature_extractor(self.model, self.features_nodes)

        self.pooling_op = create_reduction(self.pooling_op_name)
        self.aggregation_method_name = aggregation_method_name
        self.aggregation_method = None
        if aggregation_method_name is not None:
            self.aggregation_method = create_aggregation(aggregation_method_name, **kwargs)

        self.mus = None
        self.all_train_features = {}

    def start(self, *args, **kwargs):
        self.all_train_features = {}

    def update(self, x: Tensor, y: Tensor):
        self.device = x.device
        self.feature_extractor = self.feature_extractor.to(x.device)

        with torch.no_grad():
            features = self.feature_extractor(x)

        y = y.cpu()
        for k in features:
            features[k] = self.pooling_op(features[k]).cpu()

            if k not in self.all_train_features:
                self.all_train_features[k] = features[k]
            else:
                self.all_train_features[k] = torch.cat([self.all_train_features[k], features[k]], dim=0)

        if "targets" not in self.all_train_features:
            self.all_train_features["targets"] = y
        else:
            self.all_train_features["targets"] = torch.cat([self.all_train_features["targets"], y], dim=0)

    def end(self):
        self.mus = defaultdict(list)
        targets = self.all_train_features.pop("targets")
        unique_classes = torch.unique(targets).detach().cpu().numpy().tolist()
        for c in unique_classes:
            filt = targets == c
            if filt.sum() == 0:
                continue
            for k in self.all_train_features:
                self.mus[k].append(self.all_train_features[k][filt].to(self.device).mean(0, keepdim=True))

        for k in self.mus:
            self.mus[k] = torch.cat(self.mus[k], dim=0)

        if self.aggregation_method is not None and hasattr:
            _logger.info("Fitting aggregator %s...", self.aggregation_method_name)
            all_scores = []
            for i, k in enumerate(self.all_train_features):
                train_scores = []
                self.batch_size = self.all_train_features[k].shape[0]
                idx = 0
                for idx in range(0, self.all_train_features[k].shape[0], self.batch_size):
                    score = max_cosine_sim_layer_score(
                        self.all_train_features[k][idx : idx + self.batch_size].to(self.device),
                        self.mus[k].to(self.device),
                    )
                    train_scores.append(score)
                train_scores = torch.cat(train_scores, dim=0)
                all_scores.append(train_scores.view(-1, 1))
            all_scores = torch.cat(all_scores, dim=1)
            self.aggregation_method.fit(all_scores, targets)

        del self.all_train_features

    def __call__(self, x: Tensor):
        self.feature_extractor = self.feature_extractor.to(x.device)
        with torch.no_grad():
            features = self.feature_extractor(x)

        scores = {}
        for k in features:
            features[k] = self.pooling_op(features[k])
            scores[k] = max_cosine_sim_layer_score(features[k], self.mus[k].to(features[k].device)).view(-1, 1)

        # combine scores
        stack = torch.cat(list(scores.values()), dim=-1)
        if stack is None:
            raise ValueError("Stack is None, this should not happen.")

        if stack.shape[1] > 1 and self.aggregation_method is None:
            stack = stack.mean(1, keepdim=True)
        elif stack.shape[1] > 1 and self.aggregation_method is not None:
            stack = self.aggregation_method(stack)

        return stack
