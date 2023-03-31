import torch
from sklearn.preprocessing import QuantileTransformer
from torch import Tensor


class QuantileAggregation:
    def fit(self, stack: Tensor, *args, **kwargs):
        stack = stack.detach().cpu().numpy()
        self.method = QuantileTransformer(n_quantiles=100, output_distribution="uniform")
        self.method.fit(stack)

    def __call__(self, scores: Tensor, *args, **kwargs):
        device = scores.device
        scores = scores.detach().cpu().numpy()
        return torch.from_numpy(self.method.transform(scores).mean(1)).to(device).view(-1)
