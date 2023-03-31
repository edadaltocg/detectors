import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from torch import Tensor


class AnomalyBaseAggregation:
    def __init__(self, method_class, **kwargs) -> None:
        self.method = method_class(**kwargs)

    def fit(self, stack: Tensor, *args, **kwargs):
        stack = stack.detach().cpu().numpy()
        self.method.fit(stack)

    def __call__(self, scores: Tensor, *args, **kwargs):
        device = scores.device
        scores = scores.detach().cpu().numpy()
        return torch.from_numpy(self.method.score_samples(scores)).to(device).view(-1)


class LOFAggregation(AnomalyBaseAggregation):
    def __init__(self, n_neighbors: int = 20, p=2, **kwargs) -> None:
        super().__init__(LocalOutlierFactor, n_neighbors=n_neighbors, p=p, **kwargs)


class IFAggregation(AnomalyBaseAggregation):
    def __init__(self, n_estimators=100, **kwargs) -> None:
        super().__init__(IsolationForest, n_estimators=n_estimators, **kwargs)
