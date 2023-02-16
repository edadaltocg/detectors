import timm
import timm.models
import torch
import torch.nn as nn
from timm.models import register_model as timm_register_model

from detectors.models.utils import ModelDefaultConfig


def _cfg(url, architecture: str, **kwargs):
    model = timm.create_model(architecture, pretrained=False)
    base_cfg = model.default_cfg
    num_classes = kwargs.pop("num_classes", 10)
    # replace base cfg with new arguments
    base_cfg.update(kwargs)
    base_cfg["architecture"] = architecture
    base_cfg["url"] = url
    base_cfg["num_classes"] = num_classes
    return ModelDefaultConfig(**base_cfg)


default_cfgs = {
    "vit_base_patch16_224_in21k_ft_cifar10": _cfg(url="", architecture="timm/vit_base_patch16_224.orig_in21k_ft_in1k"),
    "vit_base_patch16_224_in21k_ft_cifar100": _cfg(
        url="", num_classes=100, architecture="timm/vit_base_patch16_224.orig_in21k_ft_in1k"
    ),
    "vit_base_patch16_224_in21k_ft_svhn": _cfg(url="", architecture="timm/vit_base_patch16_224.orig_in21k_ft_in1k"),
}


def _create_vit_ft(variant, pretrained=False, **kwargs):
    default_cfg = default_cfgs[variant]

    # load timm model
    model = timm.create_model(default_cfg["architecture"], pretrained=True)
    # override timm config
    model.default_cfg = default_cfg

    # override model
    model.head = nn.Linear(model.head.in_features, default_cfg["num_classes"])

    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(default_cfg["url"]))

    return model


@timm_register_model
def vit_base_patch16_224_in21k_ft_cifar10(pretrained=False, **kwargs):
    return _create_vit_ft("vit_base_patch16_224_in21k_ft_cifar10", pretrained=pretrained, **kwargs)


@timm_register_model
def vit_base_patch16_224_in21k_ft_cifar100(pretrained=False, **kwargs):
    return _create_vit_ft("vit_base_patch16_224_in21k_ft_cifar100", pretrained=pretrained, **kwargs)


@timm_register_model
def vit_base_patch16_224_in21k_ft_svhn(pretrained=False, **kwargs):
    return _create_vit_ft("vit_base_patch16_224_in21k_ft_svhn", pretrained=pretrained, **kwargs)
