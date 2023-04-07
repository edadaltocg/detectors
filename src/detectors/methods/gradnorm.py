from typing import Optional

import torch
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor

HYPERPARAMETERS = dict(temperature=dict(low=0.1, high=1000, step=0.1))


def gradnorm(x: Tensor, model: nn.Module, last_layer_name: Optional[str] = None, temperature: float = 1.0, **kwargs):
    """GradNorm OOD detector.

    Args:
        x (Tensor): input tensor.
        model (nn.Module): classifier.
        last_layer_name (Optional[str], optional): last layer node name. Defaults to None.
            If None, the last layer is automatically selected.
        temperature (float, optional): softmax temperature parameter. Defaults to 1.0.

    Returns:
        Tensor: scores for each input.

    References:
        [1] https://arxiv.org/abs/2110.00218
    """

    if last_layer_name is None:
        last_layer_name = list(model._modules.keys())[-1]
    last_layer = model._modules[last_layer_name]
    assert isinstance(last_layer, nn.Linear), "Last layer must be a linear layer"
    # feature extractor
    penultimate_layer_name = list(model._modules.keys())[-2]
    feature_extractor = create_feature_extractor(model, [penultimate_layer_name])
    with torch.no_grad():
        features = feature_extractor(x)[penultimate_layer_name]

    scores = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    for i, l in enumerate(features):
        l = l.unsqueeze(0)
        last_layer.zero_grad()
        loss = torch.mean(torch.sum(-torch.log_softmax(last_layer(l) / temperature, dim=-1), dim=-1))
        loss.backward()
        layer_grad_norm = torch.sum(torch.abs(last_layer.weight.grad.data))
        scores[i] = layer_grad_norm
    return -scores
