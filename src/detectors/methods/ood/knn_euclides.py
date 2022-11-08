import torch
from torch import Tensor, nn


class KnnEuclides:
    def __init__(self, feature_extractor: nn.Module, alpha: float = 1, k: int = 100, *args, **kwargs):
        self.feature_extractor = feature_extractor

        self.alpha = alpha
        self.k = k
        self.X = None

        assert 0 < self.alpha <= 1, "alpha must be in the interval (0, 1]"

    def on_fit_start(self, *args, **kwargs):
        self.X = None
        pass

    def fit(self, x: Tensor, *args, **kwargs):
        with torch.no_grad():
            features = self.feature_extractor(x)

        if self.X is None:
            self.X = features
        else:
            self.X = torch.cat((self.X, features), dim=0)

    def on_fit_end(self, *args, **kwargs):
        if self.X is None:
            raise ValueError(f"You must properly fit {self.__class__.__name__ } method first.")

        # random choice of alpha percent of X
        self.X = self.X[torch.randperm(self.X.shape[0])[: int(self.alpha * self.X.shape[0])]]
        self.X = self.X / torch.norm(self.X, p=2, dim=-1, keepdim=True)  # type: ignore

    def __call__(self, x: Tensor, *args, **kwargs):
        if self.X is None:
            raise ValueError(f"You must properly fit {self.__class__.__name__ } method first.")

        with torch.no_grad():
            features = self.feature_extractor(x)

        # normalize test features
        features = features / torch.norm(features, p=2, dim=-1, keepdim=True)  # type: ignore

        # pairwise euclidean distance between x and X
        dist = torch.cdist(features, self.X)

        # take smallest k distance for each test sample
        topk = torch.topk(dist, k=self.k, dim=-1, largest=False).values

        # return mean of top k distances
        return -topk.mean(-1)


def test():
    import torchvision

    x = torch.randn(10, 3, 224, 224)
    model = torchvision.models.resnet18(pretrained=True)
    knn_euclides = KnnEuclides(model, alpha=1, k=5)
    knn_euclides.on_fit_start()
    knn_euclides.fit(x)
    knn_euclides.on_fit_end()
    print(knn_euclides(x))

    assert knn_euclides(x).shape == (len(x),)


if __name__ == "__main__":
    test()
