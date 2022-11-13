from collections import defaultdict
from typing import List
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm


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


def projection_layer_score(x: Tensor, mus: Tensor):
    stack = torch.zeros(x.shape[0], len(mus), device=x.device, dtype=x.dtype)
    for i, mu in enumerate(mus):
        stack[:, i] = F.cosine_similarity(x, mu.unsqueeze(0), dim=-1)
    return torch.norm(x, p=2, dim=-1, keepdim=True) * stack  # type: ignore


class Projection:
    def __init__(
        self,
        model: torch.nn.Module,
        features_nodes: List[str],
        pooling_name: str = "max",
        aggregation_method=None,
        *args,
        **kwargs
    ):
        self.model = model
        self.name = pooling_name
        self.features_nodes = features_nodes
        self.feature_extractor = create_feature_extractor(model, features_nodes)
        self.pooling_op = reductions_registry[pooling_name]
        self.aggregation_method = aggregation_method
        self.n_classes = None
        self.mus = None
        self.max_trajectory = None
        self.ref_trajectory = None
        self.scale = 1.0

        self.all_train_features = {}

    def on_fit_start(self, *args, **kwargs):
        self.all_train_features = {}

    def fit(self, x: Tensor, y: Tensor):
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

    def on_fit_end(self):
        self.mus = defaultdict(list)
        targets = self.all_train_features.pop("targets")
        unique_classes = torch.unique(targets).detach().cpu().numpy().tolist()
        print(len(unique_classes))
        for c in tqdm(unique_classes):
            filt = targets == c
            if filt.sum() == 0:
                continue
            for k in self.all_train_features:
                self.mus[k].append(self.all_train_features[k][filt].mean(0, keepdim=True))

        for k in self.mus:
            self.mus[k] = torch.cat(self.mus[k], dim=0)
        print(self.mus[k].shape)
        # build trajectories
        logits = self.all_train_features[list(self.all_train_features.keys())[-1]]
        probs = torch.softmax(logits, 1)
        trajectories = {}
        for k in self.mus:
            trajectories[k] = projection_layer_score(self.all_train_features[k], self.mus[k])
            print(trajectories[k].shape, probs.shape)
            trajectories[k] = torch.sum(trajectories[k] * probs, 1, keepdim=True)

        trajectories = torch.cat(list(trajectories.values()), dim=-1)

        self.max_trajectory = trajectories.max(dim=0, keepdim=True)[0]
        self.ref_trajectory = trajectories.mean(dim=0, keepdim=True) / self.max_trajectory
        self.scale = torch.sum(self.ref_trajectory**2)  # type: ignore

        del self.all_train_features

    def __call__(self, x: Tensor):
        with torch.no_grad():
            features = self.feature_extractor(x)

        logits = features[list(features.keys())[-1]]
        probs = torch.softmax(logits, 1)
        scores = {}
        for k in features:
            features[k] = self.pooling_op(features[k])
            scores[k] = projection_layer_score(features[k], self.mus[k].to(features[k].device))  # type: ignore
            scores[k] = torch.sum(scores[k] * probs, 1, keepdim=True)
            print(scores[k].device)

        # combine scores
        scores = torch.cat(list(scores.values()), dim=-1)
        scores = scores / self.max_trajectory
        scores = torch.sum(scores * self.ref_trajectory, -1) / self.scale  # type: ignore

        return scores


def test():
    import torchvision.models as models

    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 10)
    model.eval()
    x = torch.rand(32, 3, 224, 224)
    y = torch.randint(0, 10, (32,))
    projection = Projection(model, ["layer4", "fc"], "max")
    projection.fit(x, y)
    projection.on_fit_end()
    print(projection(x))
    assert projection(x).shape == (32,)


if __name__ == "__main__":
    test()
