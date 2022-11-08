from functools import reduce
from typing import List

import torch
import torch.utils.data
import torchvision.models as models
from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


def get_composed_attr(model, attrs: List[str]):
    return reduce(lambda x, y: getattr(x, y), attrs, model)


class Dice:
    def __init__(
        self,
        model: torch.nn.Module,
        penultimate_node: str = "flatten",
        last_layer_name: str = "fc",
        p=0.7,
        *args,
        **kwargs
    ) -> None:
        self.model = model
        self.penultimate_node = penultimate_node
        self.feature_extractor = create_feature_extractor(model, [penultimate_node])
        self.p = p

        self.last_layer_name = last_layer_name
        last_layer_nodes = self.last_layer_name.split(".")
        self.last_layer = get_composed_attr(self.model, last_layer_nodes)

        self.weight = self.last_layer.weight.detach().clone()
        self.bias = self.last_layer.bias.detach().clone()

        m = self.weight.shape[1]
        top_k = int(m * (1 - self.p))

        self.mask = torch.ones_like(self.weight)
        top_k_weights = torch.topk(self.weight.abs(), top_k, dim=1)[0]
        self.mask[self.weight.abs() <= top_k_weights[:, -1].unsqueeze(1)] = 0

    def __call__(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            logits = self.feature_extractor(x)[self.penultimate_node] @ (self.weight * self.mask).T + self.bias
        return torch.logsumexp(logits, dim=-1)


def test():
    x = torch.randn(2, 3, 224, 224, requires_grad=False)
    model = models.vit_b_16(pretrained=True)
    print(model.heads.head)
    print(getattr(model, "heads"))
    print(getattr(getattr(model, "heads"), "head").weight)
    print(get_graph_node_names(model)[0])
    method = Dice(model, penultimate_node="getitem_5", last_layer_name="heads.head", p=0.7)
    scores = method(x)
    assert scores.shape == (2,)


if __name__ == "__main__":
    test()
