import logging
from typing import Optional, Union

import timm
import torchvision
from torch import nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform as timm_create_transform


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
    if hasattr(torchvision.models, model_name):
        return getattr(torchvision.models, model_name)(weights=weights, **kwargs)
    elif model_name in timm.list_models(pretrained=weights is not None):
        return timm.create_model(model_name, pretrained=weights is not None, **kwargs)
    else:
        raise ValueError(
            f"Model {model_name} not found in model registry, or torchvision.models, or in timm.list_models"
        )
    # else:
    #     logger.warning("`num_classes` is ignored when using torchvision models")
    #     return torch.hub.load("pytorch/vision", model=model_name, weights=weights, skip_validation=True, **kwargs)


def list_models():
    """
    List all models available in model registry, torchvision.models, and timm.list_models

    Returns:
        list: list of model names
    """
    return list(model_registry.keys()) + timm.list_models()


def create_transform(model):
    config = resolve_data_config({}, model=model)
    config["is_training"] = False
    transform = timm_create_transform(**config)

    return transform


from . import densenet, resnet


if __name__ == "__main__":
    print(list_models())
