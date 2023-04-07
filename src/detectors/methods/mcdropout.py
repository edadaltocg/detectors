import torch
from torch import Tensor, nn


@torch.no_grad()
def mcdropout(x: Tensor, model: nn.Module, k: int = 5, **kwargs) -> Tensor:
    """MC Dropout

    Forward-propagates the input through the model several times with activated dropout and averages the results.

    Args:
        x (Tensor): input tensor.
        model (nn.Module): classifier.
        k (int, optional): number of forward passes. Defaults to 5.

    References:
        [1] http://proceedings.mlr.press/v48/gal16.pdf
    """
    model.eval()

    has_dropout = False
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            has_dropout = True

    if not has_dropout:
        return torch.softmax(model(x), dim=1).max(dim=1)[0]

    results = None
    for i in range(k):
        probs = torch.softmax(model(x), dim=1)
        if results is None:
            results = probs.clone()
        results += probs
    results = results / k  # type: ignore

    return results.max(dim=1)[0]
