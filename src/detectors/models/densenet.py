"""Densenet models for CIFAR10, CIFAR100 and SVHN datasets."""
import timm
import timm.models
import torch
import torch.nn as nn
from timm.models import register_model as timm_register_model
from timm.models.densenet import _create_densenet

from detectors.data import CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD
from detectors.data.constants import CIFAR100_DEFAULT_MEAN, CIFAR100_DEFAULT_STD, SVHN_DEFAULT_MEAN, SVHN_DEFAULT_STD
from detectors.models.utils import ModelDefaultConfig, hf_hub_url_template


def _cfg(url="", **kwargs):
    num_classes = kwargs.pop("num_classes", 10)
    mean = kwargs.pop("mean", CIFAR10_DEFAULT_MEAN)
    std = kwargs.pop("std", CIFAR10_DEFAULT_STD)
    return ModelDefaultConfig(
        url=url,
        num_classes=num_classes,
        input_size=(3, 32, 32),
        pool_size=(4, 4),
        crop_pct=0.875,
        interpolation="bilinear",
        mean=mean,
        std=std,
        first_conv="features.conv0",
        classifier="classifier",
        **kwargs,
    )


def _cfg_pretrained(url, architecture, **kwargs):
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
    "densenet121_cifar10": _cfg(
        url=hf_hub_url_template("densenet121_cifar10"),
        architecture="densenet121",
    ),
    "densenet121_cifar100": _cfg(
        url=hf_hub_url_template("densenet121_cifar100"),
        num_classes=100,
        mean=CIFAR100_DEFAULT_MEAN,
        std=CIFAR100_DEFAULT_STD,
        architecture="densenet121",
    ),
    "densenet121_svhn": _cfg(
        url=hf_hub_url_template("densenet121_svhn"),
        mean=SVHN_DEFAULT_MEAN,
        std=SVHN_DEFAULT_STD,
        architecture="densenet121",
    ),
    "densenet121_camelyon17": _cfg_pretrained(
        url=hf_hub_url_template("densenet121_camelyon17"),
        architecture="densenet121",
        num_classes=2,
        input_size=(3, 96, 96),
    ),
}


def _create_densenet_small(variant, block_config, pretrained=False, **kwargs):
    default_cfg = default_cfgs[variant]

    # load timm model
    architecture = default_cfg.architecture
    model = _create_densenet(architecture, growth_rate=12, block_config=block_config, pretrained=False, **kwargs)

    # override timm config
    model.default_cfg = default_cfg
    model.pretrained_cfg = default_cfg

    # override model
    model.features.conv0 = nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False)
    model.features.norm0 = nn.Identity()
    model.features.pool0 = nn.Identity()
    model.classifier = nn.Linear(model.classifier.in_features, model.default_cfg.num_classes)

    # load weights
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(model.default_cfg.url, map_location="cpu", file_name=f"{variant}.pth")
        )

    return model


def _create_densenet_pretrained(variant, pretrained=False, **kwargs):
    default_cfg = default_cfgs[variant]

    # load timm model
    model = timm.create_model(default_cfg["architecture"], pretrained=not pretrained, **kwargs)
    model.default_cfg = default_cfg
    model.classifier = nn.Linear(model.classifier.in_features, model.default_cfg["num_classes"])

    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(default_cfg["url"], map_location="cpu", file_name=f"{variant}.pth")
        )

    return model


@timm_register_model
def densenet121_cifar10(pretrained=False, **kwargs):
    return _create_densenet_small("densenet121_cifar10", block_config=(6, 12, 24, 16), pretrained=pretrained, **kwargs)


@timm_register_model
def densenet121_cifar100(pretrained=False, **kwargs):
    return _create_densenet_small("densenet121_cifar100", block_config=(6, 12, 24, 16), pretrained=pretrained, **kwargs)


@timm_register_model
def densenet121_svhn(pretrained=False, **kwargs):
    return _create_densenet_small("densenet121_svhn", block_config=(6, 12, 24, 16), pretrained=pretrained, **kwargs)


@timm_register_model
def densenet121_camelyon17(pretrained=False, **kwargs):
    return _create_densenet_pretrained("densenet121_camelyon17", pretrained=pretrained, **kwargs)