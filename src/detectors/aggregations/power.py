import torch
from sklearn.preprocessing import PowerTransformer
from torch import Tensor


class PowerAggregation:
    def fit(self, stack: Tensor, *args, **kwargs):
        stack = stack.detach().cpu().numpy()
        self.method = PowerTransformer()
        self.method.fit(stack)

    def __call__(self, scores: Tensor, *args, **kwargs):
        device = scores.device
        scores = scores.detach().cpu().numpy()
        return torch.from_numpy(self.method.transform(scores).mean(1)).to(device).view(-1)
