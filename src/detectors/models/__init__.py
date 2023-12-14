import timm
import timm.data
import torch
from timm import create_model, list_models
from timm.models import register_model as timm_register_model

from .densenet import *
from .resnet import *
from .utils import *
from .vgg import *
from .vit import *
from .dino import *


def create_transform(model: torch.nn.Module, is_training: bool = False):
    """Create a input transformation for a given model.

    Based on the default configuration of the model following timm's library.

    Args:
        model (torch.nn.Module): Model to create the transformation for.
        is_training (bool, optional): Whether the transformation is for training or not. Defaults to False.

    Returns:
        Callable: The transformation.
    """
    data_config = timm.data.resolve_data_config(model.default_cfg)
    data_config["is_training"] = is_training
    transform = timm.data.create_transform(**data_config)
    return transform