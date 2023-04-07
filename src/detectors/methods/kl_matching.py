import torch
from torch import Tensor, nn


def kl_divergence(p: Tensor, q: Tensor, eps=1e-6):
    return (p * torch.log(p / (q + eps))).sum(1)


def js_divergence(p: Tensor, q: Tensor):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


class KLMatching:
    """KL-Matching detector.

    Args:
        model (nn.Module): classifier.

    References:
        [1] https://arxiv.org/abs/1911.11132
    """

    def __init__(self, model: nn.Module, **kwargs):
        self.model = model
        self.model.eval()

        self.centroids = {}
        self.train_probs = []
        self.predictions = []

    def start(self, *args, **kwargs):
        self.centroids = {}
        self.train_probs = []
        self.predictions = []

    @torch.no_grad()
    def update(self, x: Tensor, *args, **kwargs):
        logits = self.model(x)
        y_hat = torch.argmax(logits, dim=1).cpu()
        probs = torch.softmax(logits, dim=1).cpu()

        self.train_probs.append(probs)
        self.predictions.append(y_hat)

    def end(self):
        self.train_probs = torch.cat(self.train_probs, dim=0)
        self.predictions = torch.cat(self.predictions, dim=0)
        for c in torch.unique(self.predictions).detach().cpu().numpy().tolist():
            self.centroids[c] = torch.mean(self.train_probs[self.predictions == c], dim=0, keepdim=True)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.model(x), dim=1)
        predictions = probs.argmax(dim=1)
        scores = torch.empty_like(predictions, dtype=torch.float32)
        for label in torch.unique(predictions).detach().cpu().numpy().tolist():
            if label not in self.centroids:
                raise ValueError(f"Label {label} not found in training set.")
            centroid = self.centroids[label].to(x.device)
            scores[predictions == label] = kl_divergence(probs[predictions == label], centroid)

        return -scores
