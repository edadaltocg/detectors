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
        self.X = []
        self.targets = []

    def on_fit_start(self, *args, **kwargs):
        self.X = []
        self.targets = []

    def fit(self, X: Tensor, labels: Tensor, *args, **kwargs):
        # TODO: improve this method to accumulate centroids per class
        with torch.no_grad():
            logits = self.model(X)
        self.X.append(logits.cpu())  # type: ignore
        self.targets.append(labels.cpu())  # type: ignore

    def on_fit_end(self, *args, **kwargs):
        self.centroids = {}
        self.X = torch.cat(self.X, dim=0)
        self.targets = torch.cat(self.targets, dim=0)

        unique_classes = torch.unique(self.targets).numpy().tolist()
        for c in unique_classes:
            if torch.sum(self.targets == c).item() == 0:
                continue
            self.centroids[c] = torch.mean(self.X[self.targets == c], dim=0, keepdim=True)
        self.centroids = torch.vstack(list(self.centroids.values()))

    def __call__(self, x: Tensor):
        with torch.no_grad():
            logits = self.model(x)
        stack = igeoodlogits_vec(logits, temperature=self.temperature, centroids=self.centroids.to(logits.device))  # type: ignore
        return torch.sum(stack, dim=1, keepdim=True)


def test():
    import torchvision

    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 10, (2,))
    model = torchvision.models.resnet18(pretrained=True)
    igeood = IgeoodLogits(model)
    igeood.fit(x, y)
    igeood.on_fit_end()
    print(igeood(x))


if __name__ == "__main__":
    test()
