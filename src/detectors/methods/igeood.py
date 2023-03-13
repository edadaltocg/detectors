import torch
import torch.nn.functional as F
from torch import Tensor


def igeoodlogits_vec(logits, temperature, centroids, epsilon=1e-12, *args, **kwargs):
    logits = torch.sqrt(F.softmax(logits / temperature, dim=1))
    centroids = torch.sqrt(F.softmax(centroids / temperature, dim=1))
    mult = logits @ centroids.T
    stack = 2 * torch.acos(torch.clamp(mult, -1 + epsilon, 1 - epsilon))
    return stack


class IgeoodLogits:
    def __init__(self, model: torch.nn.Module, temperature: float = 1.0, *args, **kwargs):
        self.model = model
        self.temperature = temperature

        self.centroids = {}
        self.train_logits = []
        self.targets = []

    def start(self, *args, **kwargs):
        self.centroids = {}
        self.train_logits = []
        self.targets = []

    @torch.no_grad()
    def update(self, X: Tensor, labels: Tensor, *args, **kwargs):
        logits = self.model(X)
        self.train_logits.append(logits.cpu())  # type: ignore
        self.targets.append(labels.cpu())  # type: ignore

    def end(self, *args, **kwargs):
        self.train_logits = torch.cat(self.train_logits, dim=0)
        self.targets = torch.cat(self.targets, dim=0)

        unique_classes = torch.unique(self.targets).detach().cpu().numpy().tolist()
        for c in unique_classes:
            if torch.sum(self.targets == c).item() == 0:
                continue
            self.centroids[c] = torch.mean(self.train_logits[self.targets == c], dim=0, keepdim=True)
        self.centroids = torch.vstack(list(self.centroids.values()))

    def __call__(self, x: Tensor):
        with torch.no_grad():
            logits = self.model(x)
        stack = igeoodlogits_vec(logits, temperature=self.temperature, centroids=self.centroids.to(logits.device))  # type: ignore
        return torch.mean(stack, dim=1)
