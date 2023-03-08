"""Finetuned ViT models for CIFAR10, CIFAR100, and SVHN datasets."""
import timm
import timm.models
import torch
import torch.nn as nn
from timm.models import register_model as timm_register_model

from detectors.models.utils import ModelDefaultConfig, hf_hub_url_template


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
    "vit_base_patch16_224_in21k_ft_cifar10": _cfg(
        url=hf_hub_url_template("vit_base_patch16_224_in21k_ft_cifar10"),
        architecture="timm/vit_base_patch16_224.orig_in21k_ft_in1k",
    ),
    "vit_base_patch16_224_in21k_ft_cifar100": _cfg(
        url=hf_hub_url_template("vit_base_patch16_224_in21k_ft_cifar100"),
        num_classes=100,
        architecture="timm/vit_base_patch16_224.orig_in21k_ft_in1k",
    ),
    "vit_base_patch16_224_in21k_ft_svhn": _cfg(
        url=hf_hub_url_template("vit_base_patch16_224_in21k_ft_svhn"),
        architecture="timm/vit_base_patch16_224.orig_in21k_ft_in1k",
    ),
}

# import timm
# import torch
# from torch import nn

# model = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=False)
# model.head = nn.Linear(model.head.in_features, 10)
# model.load_state_dict(
#     torch.hub.load_state_dict_from_url(
#         "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_svhn/resolve/main/pytorch_model.bin",
#         map_location="cpu",
#         file_name="vit_base_patch16_224_in21k_ft_svhn.pth",
#     )
# )
# x = torch.randn(1, 3, 224, 224)
# y = model(x)
# assert y.shape == (1, 10)


def _create_vit_ft(variant, pretrained=False, **kwargs):
    default_cfg = default_cfgs[variant]

    # load timm model
    model = timm.create_model(default_cfg["architecture"], pretrained=not pretrained)
    # override timm config
    model.default_cfg = default_cfg

    # override model
    model.head = nn.Linear(model.head.in_features, default_cfg["num_classes"])

    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(default_cfg["url"], map_location="cpu", file_name=f"{variant}.pth")
        )

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
