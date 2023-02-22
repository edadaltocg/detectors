import logging
from functools import reduce
from typing import List, Optional

import torch
import torch.utils.data
from torch import Tensor

_logger = logging.getLogger(__name__)


def get_composed_attr(model, attrs: List[str]):
    return reduce(lambda x, y: getattr(x, y), attrs, model)


class Dice:
    """
    DICE: Leveraging Sparsification for Out-of-Distribution Detection

    Reference:
    ----------

    Sun, Y., & Li, Y. (2021). DICE: Leveraging Sparsification for Out-of-Distribution Detection. ArXiv. https://doi.org/10.48550/arXiv.2111.09805
    """

    def __init__(self, model: torch.nn.Module, last_layer_name: Optional[str] = "fc", p=0.7, *args, **kwargs) -> None:
        self.model = model
        self.p = p

        self.last_layer_name = last_layer_name
        if self.last_layer_name is None and hasattr(model, "default_cfg"):
            self.last_layer_name = model.default_cfg["classifier"]
        elif self.last_layer_name is None:
            raise ValueError("last_layer_name must be specified")

        self._weight_backup = get_composed_attr(self.model, self.last_layer_name.split(".")).weight.clone()
        self._bias_backup = get_composed_attr(self.model, self.last_layer_name.split(".")).bias.clone()

        self.last_layer_nodes = self.last_layer_name.split(".")

        weight = get_composed_attr(self.model, self.last_layer_nodes).weight.clone()

        self.m = weight.shape[1]
        self.top_k = int(self.m * (1 - self.p))
        _logger.info("Dice top_k: %s ", self.top_k)

        self.mask = torch.ones_like(weight)
        top_k_weights = torch.topk(weight, self.top_k, dim=1).values
        self.mask[weight <= top_k_weights[:, -1].unsqueeze(1)] = 0

        get_composed_attr(self.model, self.last_layer_nodes).weight.data *= self.mask.data
        _logger.info(
            (get_composed_attr(self.model, self.last_layer_nodes).weight.data - self._weight_backup.data).sum().item()
        )
        assert not torch.allclose(
            get_composed_attr(self.model, self.last_layer_nodes).weight.data, self._weight_backup.data
        )
        assert torch.allclose(get_composed_attr(self.model, self.last_layer_nodes).bias.data, self._bias_backup.data)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        return torch.logsumexp(logits, dim=-1)
