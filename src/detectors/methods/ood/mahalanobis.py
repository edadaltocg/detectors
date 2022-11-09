from typing import List

import torch
from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


def mahalanobis_distance_inv(x: Tensor, y: Tensor, inverse: Tensor):
    return torch.nan_to_num(torch.sqrt(((x - y).T * (inverse @ (x - y).T)).sum(0)), 1e9)


def mahalanobis_inv_layer_score(x: Tensor, mus: Tensor, inv: Tensor):
    stack = torch.zeros((x.shape[0], mus.shape[0]), device=x.device, dtype=torch.float32)
    for i, mu in enumerate(mus):
        stack[:, i] = mahalanobis_distance_inv(x, mu.reshape(1, -1), inv).reshape(-1)

    return -stack.min(1, keepdim=True)[0]


def torch_reduction_matrix(sigma: Tensor, reduction_method="pseudo"):

    if reduction_method == "cholesky":
        C = torch.linalg.cholesky(sigma)
        return torch.linalg.inv(C.T)
    elif reduction_method == "svd":
        u, s, _ = torch.linalg.svd(sigma)
        return u @ torch.diag(torch.sqrt(1 / s))
    elif reduction_method == "pseudo":
        return torch.linalg.pinv(sigma)


def class_cond_mus_cov_inv_matrix(x: Tensor, targets: Tensor, inv_method="pseudo"):
    unique_classes = torch.unique(targets).detach().cpu().numpy().tolist()
    class_cond_dot = {}
    class_cond_mean = {}
    for c in unique_classes:
        filt = targets == c
        temp = x[filt]
        class_cond_dot[c] = torch.cov(temp.T)
        class_cond_mean[c] = temp.mean(0, keepdim=True)
    cov_mat = sum(list(class_cond_dot.values())) / x.shape[0]
    inv_mat = torch_reduction_matrix(cov_mat, reduction_method=inv_method)
    mus = torch.vstack(list(class_cond_mean.values()))
    return mus, cov_mat, inv_mat


class Mahalanobis:
    def __init__(
        self,
        model: torch.nn.Module,
        features_nodes: List[str],
        reduction_method: str = "pseudo",
        aggregation_method=None,
        *args,
        **kwargs
    ) -> None:
        self.model = model
        self.feature_extractor = create_feature_extractor(model, features_nodes)
        self.reduction_method = reduction_method
        self.aggregation_method = aggregation_method

        self.mus = []
        self.invs = []

        self.all_training_features = {}

    def on_fit_start(self, *args, **kwargs):
        self.all_training_features = {}

    def fit(self, x: Tensor, y: Tensor) -> None:
        # TODO: make accumulation of features more efficient (sum and sum squared)
        with torch.no_grad():
            features = self.feature_extractor(x)

        # accumulate training features
        if len(self.all_training_features) == 0:
            for k in features:
                self.all_training_features[k] = features[k]
            self.all_training_features["targets"] = y
        else:
            for k in features:
                self.all_training_features[k] = torch.cat((self.all_training_features[k], features[k]), dim=0)
            self.all_training_features["targets"] = torch.cat((self.all_training_features["targets"], y), dim=0)

    def on_fit_end(self, *args, **kwargs):
        for k in self.all_training_features:
            if k == "targets":
                continue
            mu, cov, inv = class_cond_mus_cov_inv_matrix(
                self.all_training_features[k], self.all_training_features["targets"], self.reduction_method
            )
            self.mus.append(mu)
            self.invs.append(inv)

    def __call__(self, x: Tensor) -> Tensor:
        if len(self.invs) == 0 or len(self.mus) == 0:
            raise ValueError("You must properly fit the Mahalanobis method first.")

        with torch.no_grad():
            features = self.feature_extractor(x)

        features_keys = list(features.keys())
        stack = None
        for i, (mu, inv) in enumerate(zip(self.mus, self.invs)):
            scores = mahalanobis_inv_layer_score(features[features_keys[i]], mu, inv)
            if i == 0:
                stack = scores
            else:
                stack = torch.cat((stack, scores), dim=1)  # type: ignore
        if stack is None:
            raise ValueError("Stack is None, this should not happen.")

        if stack.shape[1] > 1 and self.aggregation_method is None:
            stack = stack.mean(1, keepdim=True)
        elif stack.shape[1] > 1 and self.aggregation_method is not None:
            # TODO: validation combine stack
            stack = self.aggregation_method(stack)

        return stack
