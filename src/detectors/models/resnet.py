import timm
import timm.models
import torch
import torch.nn as nn
from timm.models import register_model as timm_register_model

from detectors.data import CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD
from detectors.data.constants import CIFAR100_DEFAULT_MEAN, CIFAR100_DEFAULT_STD, SVHN_DEFAULT_MEAN, SVHN_DEFAULT_STD
from detectors.models.utils import ModelDefaultConfig, hf_hub_url_template
import logging


_logger = logging.getLogger(__name__)


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
        first_conv="conv1",
        classifier="fc",
        **kwargs,
    )


default_cfgs = {
    "resnet18_cifar10": _cfg(url=hf_hub_url_template("resnet18_cifar10"), architecture="resnet18"),
    "resnet34_cifar10": _cfg(url=hf_hub_url_template("resnet34_cifar10"), architecture="resnet34"),
    "resnet50_cifar10": _cfg(url=hf_hub_url_template("resnet50_cifar10"), architecture="resnet50"),
    "resnet18_cifar100": _cfg(
        url=hf_hub_url_template("resnet18_cifar100"),
        num_classes=100,
        mean=CIFAR100_DEFAULT_MEAN,
        std=CIFAR100_DEFAULT_STD,
        architecture="resnet18",
    ),
    "resnet34_cifar100": _cfg(
        url=hf_hub_url_template("resnet34_cifar100"),
        num_classes=100,
        mean=CIFAR100_DEFAULT_MEAN,
        std=CIFAR100_DEFAULT_STD,
        architecture="resnet34",
    ),
    "resnet50_cifar100": _cfg(
        url=hf_hub_url_template("resnet50_cifar100"),
        num_classes=100,
        mean=CIFAR100_DEFAULT_MEAN,
        std=CIFAR100_DEFAULT_STD,
        architecture="resnet50",
    ),
    "resnet18_svhn": _cfg(
        url=hf_hub_url_template("resnet18_svhn"), mean=SVHN_DEFAULT_MEAN, std=SVHN_DEFAULT_STD, architecture="resnet18"
    ),
    "resnet34_svhn": _cfg(
        url=hf_hub_url_template("resnet34_svhn"), mean=SVHN_DEFAULT_MEAN, std=SVHN_DEFAULT_STD, architecture="resnet34"
    ),
    "resnet50_svhn": _cfg(
        url=hf_hub_url_template("resnet50_svhn"), mean=SVHN_DEFAULT_MEAN, std=SVHN_DEFAULT_STD, architecture="resnet50"
    ),
}


def _create_resnet_small(variant, features_dim=512, pretrained=False, **kwargs):
    default_cfg = default_cfgs[variant]

    # load timm model
    architecture = default_cfg.architecture or variant.split("_")[0]
    model = timm.create_model(architecture, pretrained=False)

    # override timm config
    model.default_cfg = default_cfg
    model.pretrained_cfg = default_cfg

    # override model
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()  # type: ignore
    model.fc = nn.Linear(features_dim, model.default_cfg.num_classes)

    # load weights
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(model.default_cfg.url, map_location="cpu", file_name=f"{variant}.pth")
        )

    return model


@timm_register_model
def resnet18_cifar10(pretrained=False, **kwargs):
    return _create_resnet_small("resnet18_cifar10", features_dim=512, pretrained=pretrained, **kwargs)


@timm_register_model
def resnet34_cifar10(pretrained=False, **kwargs):
    return _create_resnet_small("resnet34_cifar10", features_dim=512, pretrained=pretrained, **kwargs)


@timm_register_model
def resnet50_cifar10(pretrained=False, **kwargs):
    return _create_resnet_small("resnet50_cifar10", features_dim=2048, pretrained=pretrained, **kwargs)


@timm_register_model
def resnet18_cifar100(pretrained=False, **kwargs):
    return _create_resnet_small("resnet18_cifar100", features_dim=512, pretrained=pretrained, **kwargs)


@timm_register_model
def resnet34_cifar100(pretrained=False, **kwargs):
    return _create_resnet_small("resnet34_cifar100", features_dim=512, pretrained=pretrained, **kwargs)


@timm_register_model
def resnet50_cifar100(pretrained=False, **kwargs):
    return _create_resnet_small("resnet50_cifar100", features_dim=2048, pretrained=pretrained, **kwargs)


@timm_register_model
def resnet18_svhn(pretrained=False, **kwargs):
    return _create_resnet_small("resnet18_svhn", features_dim=512, pretrained=pretrained, **kwargs)


@timm_register_model
def resnet34_svhn(pretrained=False, **kwargs):
    return _create_resnet_small("resnet34_svhn", features_dim=512, pretrained=pretrained, **kwargs)


@timm_register_model
def resnet50_svhn(pretrained=False, **kwargs):
    return _create_resnet_small("resnet50_svhn", features_dim=2048, pretrained=pretrained, **kwargs)
