import timm
import timm.models
import torch
import torch.nn as nn
from timm.models import register_model as timm_register_model

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
        crop_pct=1,
        interpolation="bilinear",
        mean=mean,
        std=std,
        first_conv="features.0",
        classifier="head.fc",
        **kwargs,
    )


default_cfgs = {
    "vgg16_bn_cifar10": _cfg(
        url=hf_hub_url_template("vgg16_bn_cifar10"),
        architecture="vgg16_bn",
    ),
    "vgg16_bn_cifar100": _cfg(
        url=hf_hub_url_template("vgg16_bn_cifar100"),
        num_classes=100,
        mean=CIFAR100_DEFAULT_MEAN,
        std=CIFAR100_DEFAULT_STD,
        architecture="vgg16_bn",
    ),
    "vgg16_bn_svhn": _cfg(
        url=hf_hub_url_template("vgg16_bn_svhn"),
        mean=SVHN_DEFAULT_MEAN,
        std=SVHN_DEFAULT_STD,
        architecture="vgg16_bn",
    ),
}


def _create_vgg_small(variant, features_dim=512, pretrained=False, **kwargs):
    default_cfg = default_cfgs[variant]

    # load timm model
    architecture = default_cfg.architecture
    model = timm.create_model(architecture, pretrained=False)

    # override timm config
    model.default_cfg = default_cfg
    model.pretrained_cfg = default_cfg

    # override model
    model.pre_logits = nn.Identity()  # type: ignore
    model.head.fc = nn.Linear(features_dim, model.default_cfg.num_classes)

    # load weights
    if pretrained:
        checkpoint = model.default_cfg.url
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                checkpoint, progress=True, map_location="cpu", file_name=f"{variant}.pth"
            )
        )

    return model


@timm_register_model
def vgg16_bn_cifar10(pretrained=False, **kwargs):
    return _create_vgg_small("vgg16_bn_cifar10", pretrained=pretrained, **kwargs)


@timm_register_model
def vgg16_bn_cifar100(pretrained=False, **kwargs):
    return _create_vgg_small("vgg16_bn_cifar100", pretrained=pretrained, **kwargs)


@timm_register_model
def vgg16_bn_svhn(pretrained=False, **kwargs):
    return _create_vgg_small("vgg16_bn_svhn", pretrained=pretrained, **kwargs)
