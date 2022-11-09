from typing import List

import torch
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


class KnnEuclides:
    def __init__(
        self,
        model: nn.Module,
        features_nodes: List[str],
        alpha: float = 1,
        k: int = 10,
        aggregation_method=None,
        *args,
        **kwargs
    ):
        self.model = model
        self.feature_nodes = features_nodes
        self.feature_extractor = create_feature_extractor(model, features_nodes)

        self.alpha = alpha
        self.k = k
        self.aggregation_method = aggregation_method

        self.X = None

        assert 0 < self.alpha <= 1, "alpha must be in the interval (0, 1]"

    def on_fit_start(self, *args, **kwargs):
        self.X = None

    def fit(self, x: Tensor, *args, **kwargs):
        with torch.no_grad():
            features = self.feature_extractor(x)

        for k in features:
            features[k] = torch.flatten(features[k], start_dim=1)
            features[k] = features[k].cpu()

        if self.X is None:
            self.X = features
        else:
            for k in features:
                self.X[k] = torch.cat((self.X[k], features[k]), dim=0)

    def on_fit_end(self, *args, **kwargs):
        if self.X is None:
            raise ValueError(f"You must properly fit {self.__class__.__name__ } method first.")

        # random choice of alpha percent of X
        for k in self.X:
            self.X[k] = self.X[k][torch.randperm(self.X[k].shape[0])[: int(self.alpha * self.X[k].shape[0])]]
            self.X[k] = self.X[k] / torch.norm(self.X[k], p=2, dim=-1, keepdim=True)  # type: ignore

    def __call__(self, x: Tensor):
        if self.X is None:
            raise ValueError(f"You must properly fit {self.__class__.__name__ } method first.")

        with torch.no_grad():
            features = self.feature_extractor(x)

        # normalize test features
        for k in features:
            features[k] = torch.flatten(features[k], start_dim=1)
            features[k] = features[k] / torch.norm(features[k], p=2, dim=-1, keepdim=True)  # type: ignore

        # pairwise euclidean distance between x and X
        stack = []
        for k in features:
            features[k] = torch.cdist(features[k], self.X[k].to(features[k].device), p=2)
            # take smallest k distance for each test sample
            topk, _ = torch.topk(features[k], k=self.k, dim=-1, largest=False)
            stack.append(-topk[:, -1].view(-1, 1))
            # stack.append(-topk.mean(dim=-1, keepdim=True))

        stack = torch.cat(stack, dim=1)

        if stack.shape[1] > 1 and self.aggregation_method is None:
            stack = stack.mean(1, keepdim=True)
        elif stack.shape[1] > 1 and self.aggregation_method is not None:
            # TODO: validation combine stack
            stack = self.aggregation_method(stack)

        return stack.view(-1)


def test():
    import torchvision

    x = torch.randn(10, 3, 224, 224)
    model = torchvision.models.resnet18(pretrained=True)
    print(get_graph_node_names(model)[0])
    knn_euclides = KnnEuclides(model, features_nodes=["flatten"], alpha=1, k=2)
    knn_euclides.on_fit_start()
    knn_euclides.fit(x)
    knn_euclides.on_fit_end()
    print(knn_euclides(x))

    assert knn_euclides(x).shape == (len(x),)


if __name__ == "__main__":
    test()
