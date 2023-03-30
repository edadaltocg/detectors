import logging
from typing import List, Optional

import torch
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

_logger = logging.getLogger(__name__)


class KnnEuclides:
    def __init__(
        self,
        model: nn.Module,
        features_nodes: Optional[List[str]] = None,
        alpha: float = 1,
        k: int = 10,
        aggregation_method=None,
        avg_topk: bool = False,
        *args,
        **kwargs,
    ):
        self.model = model
        self.features_nodes = features_nodes
        if self.features_nodes is None:
            self.features_nodes = [list(self.model._modules.keys())[-2]]
        self.feature_extractor = create_feature_extractor(model, self.features_nodes)

        self.alpha = alpha
        self.k = k
        self.aggregation_method_name = aggregation_method_name
        self.aggregation_method = None
        if aggregation_method_name is not None:
            self.aggregation_method = create_aggregation(aggregation_method_name, **kwargs)

        self.train_features = None
        self.mean_op = avg_topk

        assert 0 < self.alpha <= 1, "alpha must be in the interval (0, 1]"

    def start(self, *args, **kwargs):
        self.train_features = None

    @torch.no_grad()
    def update(self, x: Tensor, *args, **kwargs):
        self.feature_extractor = self.feature_extractor.to(x.device)
        features = self.feature_extractor(x)

        for k in features:
            if features[k].dim() == 4:
                features[k] = torch.mean(features[k], dim=[2, 3])
            features[k] = features[k].flatten(1).cpu()

        if self.train_features is None:
            self.train_features = features
        else:
            for k in features:
                self.train_features[k] = torch.cat((self.train_features[k], features[k]), dim=0)

    def end(self, *args, **kwargs):
        if self.train_features is None:
            raise ValueError(f"You must properly fit {self.__class__.__name__ } method first.")

        # random choice of alpha percent of X
        for k in self.train_features:
            self.train_features[k] = self.train_features[k][
                torch.randperm(self.train_features[k].shape[0])[: int(self.alpha * self.train_features[k].shape[0])]
            ]
            # normalize train features
            self.train_features[k] = self.train_features[k] / torch.norm(self.train_features[k], p=2, dim=-1, keepdim=True)  # type: ignore

    @torch.no_grad()
    def __call__(self, x: Tensor):
        self.feature_extractor = self.feature_extractor.to(x.device)
        if self.train_features is None:
            raise ValueError(f"You must properly fit {self.__class__.__name__ } method first.")

        features = self.feature_extractor(x)

        # normalize test features
        for k in features:
            features[k] = torch.flatten(features[k], start_dim=1)
            features[k] = features[k] / torch.norm(features[k], p=2, dim=-1, keepdim=True)  # type: ignore

        # pairwise euclidean distance between x and X
        stack = torch.zeros((x.shape[0], len(features)), device=x.device)
        for i, k in enumerate(features):
            features[k] = torch.cdist(features[k], self.train_features[k].to(x.device), p=2)
            # take smallest k distance for each test sample
            topk, _ = torch.topk(features[k], k=self.k, dim=-1, largest=False)
            if self.mean_op:
                stack[:, i] = -topk.mean(dim=-1)
            else:
                stack[:, i] = -topk[:, -1]

        if stack.shape[1] > 1 and self.aggregation_method is None:
            stack = stack.mean(1, keepdim=True)
        elif stack.shape[1] > 1 and self.aggregation_method is not None:
            stack = self.aggregation_method(stack)

        return stack.view(-1)
