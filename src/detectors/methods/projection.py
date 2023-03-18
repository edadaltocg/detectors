import logging
from collections import defaultdict
from typing import List, Union

import torch
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

_logger = logging.getLogger(__name__)


def flatten(data: Tensor, *args, **kwargs):
    return torch.flatten(data, 1)


def adaptive_avg_pool2d(data: Tensor, *args, **kwargs):
    if len(data.shape) > 2:
        return torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(data), 1)
    return data


def adaptive_max_pool2d(data: Tensor, *args, **kwargs):
    if len(data.shape) > 2:
        return torch.flatten(nn.AdaptiveMaxPool2d((1, 1))(data), 1)
    return data


def getitem(data: Tensor, *args, **kwargs):
    return data[:, 0].clone().contiguous()


def none_reduction(data: Tensor, *args, **kwargs):
    return data


reductions_registry = {
    "flatten": flatten,
    "avg": adaptive_avg_pool2d,
    "max": adaptive_max_pool2d,
    "getitem": getitem,
    "none": none_reduction,
}


def projection_layer_score(x: Tensor, mus: Union[Tensor, List[Tensor]], eps=1e-7):
    if isinstance(mus, list):
        mus = torch.cat(mus, dim=0)
    den = torch.norm(x, dim=-1, keepdim=True) @ torch.norm(mus, dim=-1, keepdim=True).T
    stack = x @ mus.T
    stack = stack / (den + eps)
    return torch.norm(x, p=2, dim=-1, keepdim=True) * stack  # type: ignore


class Projection:
    def __init__(
        self,
        model: torch.nn.Module,
        features_nodes: List[str],
        pooling_name: str = "max",
        aggregation_method=None,
        *args,
        **kwargs,
    ):
        self.model = model
        self.model.eval()
        self.pooling_name = pooling_name
        self.device = next(self.model.parameters()).device
        self.features_nodes = features_nodes
        self.feature_extractor = create_feature_extractor(self.model, features_nodes)
        self.pooling_op = reductions_registry[pooling_name]
        self.aggregation_method = aggregation_method
        self.n_classes = None
        self.mus = None
        self.max_trajectory = None
        self.ref_trajectory = None
        self.scale = 1.0

        self.all_train_features = {}

    def start(self, *args, **kwargs):
        self.all_train_features = {}
        self.max_trajectory = None
        self.ref_trajectory = None
        self.scale = None

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

        # build trajectories
        logits = self.all_train_features[list(self.all_train_features.keys())[-1]]
        probs = torch.softmax(logits, 1).to(self.device)
        trajectories = {}
        for k in self.mus:
            trajectories[k] = projection_layer_score(self.all_train_features[k].to(self.device), self.mus[k])
            trajectories[k] = torch.sum(trajectories[k] * probs, 1, keepdim=True)

        trajectories = torch.cat(list(trajectories.values()), dim=-1)

        self.max_trajectory = trajectories.max(dim=0, keepdim=True)[0]
        self.ref_trajectory = trajectories.mean(dim=0, keepdim=True) / self.max_trajectory
        self.scale = torch.sum(self.ref_trajectory**2)  # type: ignore

        del self.all_train_features

    def __call__(self, x: Tensor):
        self.feature_extractor = self.feature_extractor.to(x.device)
        with torch.no_grad():
            features = self.feature_extractor(x)

        logits = features[list(features.keys())[-1]]
        probs = torch.softmax(logits, dim=1)
        scores = {}
        for k in features:
            features[k] = self.pooling_op(features[k])
            scores[k] = projection_layer_score(features[k], self.mus[k].to(features[k].device))  # type: ignore
            scores[k] = torch.sum(scores[k] * probs, 1, keepdim=True)

        # combine scores
        scores = torch.cat(list(scores.values()), dim=-1)
        scores = scores / self.max_trajectory.to(scores.device)
        scores = torch.sum(scores * self.ref_trajectory.to(scores.device), dim=-1) / self.scale  # type: ignore

        return scores
