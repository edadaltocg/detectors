import logging
from typing import Optional, Union

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


def create_model(model_name: str, weights: Optional[Union[str, bool]] = None, **kwargs) -> nn.Module:
    """
    Create a model

    Args:
        model_name (str): name of model to instantiate
        num_classes (int, optional): _description_. Defaults to 10.
        weights (bool | str): load pretrained weights if true or load from path if str.
        checkpoint_path (str): path of checkpoint to load after model is initialized

    Returns:
        model (nn.Module):
    """

    if model_name in list(model_registry.keys()):
        return model_registry[model_name](weights, **kwargs)
    if model_name in timm.list_models(pretrained=weights is not None):
        return timm.create_model(
            model_name,
            pretrained=weights is not None,
            checkpoint_path=weights or "",
            **kwargs,
        )
    else:
        logger.warning("`num_classes` is ignored when using torchvision models")
        return torch.hub.load("pytorch/vision", model=model_name, weights=weights, **kwargs)


from . import resnet
