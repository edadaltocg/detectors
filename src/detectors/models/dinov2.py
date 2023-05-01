import timm
import torch
from timm.models import register_model as timm_register_model

from detectors.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from detectors.models.utils import ModelDefaultConfig


# def _cfg(url="", **kwargs):
#     model = timm.create_model(architecture, pretrained=False, num_classes=0)
#     base_cfg = model.default_cfg
#     num_classes = kwargs.pop("num_classes", 10)
#     # replace base cfg with new arguments
#     base_cfg.update(kwargs)
#     base_cfg["architecture"] = architecture
#     base_cfg["url"] = url
#     base_cfg["num_classes"] = num_classes
#     return ModelDefaultConfig(**base_cfg)


def _create_dinov2(variant: str):
    model = torch.hub.load("facebookresearch/dinov2", variant)
    return model


@timm_register_model
def dinov2_vitb14(pretrained=True, **kwargs):
    return _create_dinov2("dinov2_vitb14")
