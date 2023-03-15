from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from torch import Tensor


class AnomalyBaseAggregation:
    def __init__(self, method_class, **kwargs) -> None:
        self.method = method_class(**kwargs)

    def fit(self, stack: Tensor):
        stack = stack.detach().cpu().numpy()
        self.method.fit(stack)

    def predict(self, scores: Tensor):
        scores = scores.detach().cpu().numpy()
        return self.method.score_samples(scores)


class LOFAggregation(AnomalyBaseAggregation):
    def __init__(self, n_neighbors: int = 20, p=2) -> None:
        super().__init__(LocalOutlierFactor, n_neighbors=n_neighbors, p=p)


class IFAggregation(AnomalyBaseAggregation):
    def __init__(self, n_estimators=100) -> None:
        super().__init__(IsolationForest, n_estimators=n_estimators)
