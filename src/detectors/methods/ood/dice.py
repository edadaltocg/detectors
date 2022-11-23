from functools import reduce
from typing import List

import torch
import torch.utils.data
import torchvision.models as models
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


def get_composed_attr(model, attrs: List[str]):
    return reduce(lambda x, y: getattr(x, y), attrs, model)


class Dice:
    def __init__(self, model: torch.nn.Module, last_layer_name: str = "fc", p=0.7, *args, **kwargs) -> None:
        self.model = model
        self.p = p

        self.weight_backup = get_composed_attr(self.model, last_layer_name.split(".")).weight.clone()

        self.last_layer_name = last_layer_name
        self.last_layer_nodes = self.last_layer_name.split(".")

        weight = get_composed_attr(self.model, self.last_layer_nodes).weight.clone()

        m = weight.shape[1]
        top_k = int(m * (1 - self.p))
        logger.info("Dice top_k: %s ", top_k)

        self.mask = torch.ones_like(weight)
        top_k_weights = torch.topk(weight.abs(), top_k, dim=1)[0]
        self.mask[weight.abs() <= top_k_weights[:, -1].unsqueeze(1)] = 0

        get_composed_attr(self.model, self.last_layer_nodes).weight.data *= self.mask

        assert not torch.allclose(get_composed_attr(self.model, self.last_layer_nodes).weight, self.weight_backup)

    def __call__(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            logits = self.model(x)
        return torch.logsumexp(logits, dim=-1)


def test():
    x = torch.randn(2, 3, 224, 224, requires_grad=False)
    model = models.densenet121(pretrained=True)
    logits = model(x)
    energy1 = torch.logsumexp(logits, dim=-1)
    model_cp = models.densenet121(pretrained=True)
    logits = model_cp(x)
    energy2 = torch.logsumexp(logits, dim=-1)
    assert torch.allclose(energy1, energy2)
    method = Dice(model, last_layer_name="classifier", p=0.7)
    scores = method(x)
    logits = model_cp(x)
    energy = torch.logsumexp(logits, dim=-1)
    assert not torch.allclose(scores, energy)
    assert scores.shape == (2,)


if __name__ == "__main__":
    test()
    test2()
