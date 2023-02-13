import timm
import timm.models
import torch
import torch.nn as nn
from detectors.data import CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD
from detectors.data.constants import CIFAR100_DEFAULT_MEAN, CIFAR100_DEFAULT_STD
from detectors.models.utils import ModelDefaultConfig
from timm.models import register_model as timm_register_model


def _cfg(url="", **kwargs):
    return ModelDefaultConfig(
        url=url,
        num_classes=10,
        input_size=(3, 32, 32),
        pool_size=(4, 4),
        crop_pct=0.875,
        interpolation="bilinear",
        mean=CIFAR10_DEFAULT_MEAN,
        std=CIFAR10_DEFAULT_STD,
        first_conv="features.conv0",
        classifier="classifier",
        **kwargs,
    )


default_cfgs = {
    "densenet121_cifar10": _cfg(url="", architecture="densenet121"),
    "densenet121_cifar100": _cfg(
        url="", num_classes=100, mean=CIFAR100_DEFAULT_MEAN, std=CIFAR100_DEFAULT_STD, architecture="densenet121"
    ),
}


def _create_densenet_small(variant, pretrained=False, **kwargs):
    default_cfg = default_cfgs[variant]

    # load timm model
    architecture = default_cfg.architecture or variant.split("_")[0]
    model = timm.create_model(architecture, pretrained=False)

    # override timm config
    model.default_cfg = default_cfg

    # override model
    model.features.conv0 = nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False)
    del model.features.norm0
    del model.features.pool0
    model.classifier = nn.Linear(model.classifier.in_features, model.default_cfg.num_classes)

    # load weights
    if pretrained:
        checkpoint = model.default_cfg.url
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True, map_location="cpu"))

    return model


@timm_register_model
def densenet121_cifar10(pretrained=False, **kwargs):
    return _create_densenet_small("densenet121_cifar10", pretrained=pretrained, **kwargs)


@timm_register_model
def densenet121_cifar100(pretrained=False, **kwargs):
    return _create_densenet_small("densenet121_cifar100", pretrained=pretrained, **kwargs)


if __name__ == "__main__":
    model_obj_1 = timm.create_model("densenet121", pretrained=False)
    model_obj_2 = timm.create_model("densenet121_cifar10", pretrained=False)
    print(model_obj_1.features.conv0)
    print(model_obj_2.features.conv0)
