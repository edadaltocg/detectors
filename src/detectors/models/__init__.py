import logging
from typing import Optional

import timm
import torch
from torch import nn


logger = logging.getLogger(__name__)


model_registry = {}


def register_model(name: str):
    """
    Register a model

    Args:
        name (str): name of model to register

    Returns:
        Callable: decorator
    """

    def decorator(f):
        model_registry[name] = f
        return f

    return decorator


def _create_model(model_name: str, num_classes: int = 10, weights: Optional[str] = None, **kwargs):
    model = model_registry[model_name](num_classes=num_classes, **kwargs)

    if weights:
        checkpoints = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoints)

    return model


def create_model(model_name: str, num_classes: int = 10, weights: Optional[str] = None, **kwargs) -> nn.Module:
    """
    Create a model

    Args:
        model_name (str): name of model to instantiate
        num_classes (int, optional): _description_. Defaults to 10.
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized

    Returns:
        model (nn.Module):
    """

    if model_name in torch.hub.list("pytorch/vision"):
        logger.warning("`num_classes` is ignored when using torchvision models")
        return torch.hub.load("pytorch/vision", model=model_name, weights=weights, **kwargs)
    if model_name in timm.list_models(pretrained=weights is not None):
        return timm.create_model(
            model_name,
            num_classes=num_classes,
            pretrained=weights is not None,
            checkpoint_path=weights or "",
            **kwargs,
        )
    else:
        model = _create_model(model_name, num_classes, weights, **kwargs)

    return model


from . import resnet
