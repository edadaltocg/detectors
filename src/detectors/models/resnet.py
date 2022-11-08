from typing import List, Optional, Union

import torch
import torch.nn as nn
import torchvision

from . import register_model


def ResNet18Small(num_classes=10):
    model = torchvision.models.resnet18(weights=None)
    print(model.inplanes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # type: ignore
    model.avgpool = nn.AvgPool2d(4)  # type: ignore
    model.fc = nn.Linear(512, num_classes)
    return model


@register_model("resnet18_cifar10")
def resnet18_cifar10(weights=False, **kwargs):
    """
    Resnet18 model with a small `conv1` layer and a linear layer with `num_classes` outputs

    Args:
        pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    num_classes = kwargs.get("num_classes", 10)
    model = ResNet18Small(num_classes=num_classes)

    if weights:
        checkpoint = "https://github.com/edadaltocg/detectors/releases/download/weights-v0.1.0-beta/best.pth"
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True, map_location="cpu"))

    return model


def ResNet34Small(num_classes=10):
    model = torchvision.models.resnet34(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # type: ignore
    model.avgpool = nn.AvgPool2d(4)  # type: ignore
    model.fc = nn.Linear(512, num_classes)
    return model


@register_model("resnet34_cifar10")
def resnet34_cifar10(weights: Optional[Union[str, bool]] = False, **kwargs):
    """
    Resnet34 model with a small `conv1` layer and a linear layer with 10 classes outputs

    Args:
        pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    num_classes = kwargs.get("num_classes", 10)
    model = ResNet34Small(num_classes=num_classes)

    if weights:
        if isinstance(weights, str):
            checkpoint = weights
        else:
            checkpoint = (
                "https://github.com/edadaltocg/detectors/releases/download/weights-v0.1.0-beta/resnet34_cifar10_42.pth"
            )

        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True, map_location="cpu"))

    return model


def ResNet101Small(num_classes=10):
    model = torchvision.models.resnet101(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # type: ignore
    model.avgpool = nn.AvgPool2d(4)  # type: ignore
    model.fc = nn.Linear(2048, num_classes)
    return model


def get_default_features_nodes() -> List[str]:
    return []


if __name__ == "__main__":
    model = ResNet18Small()
    print(model)
    model = ResNet34Small()
    print(model)
    model = ResNet101Small()
    print(model)

    model = resnet34_cifar10(weights=False)

    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y)
    assert y.shape == (1, 10)

    model = resnet18_cifar10(weights=False)
    print("resnet18", model(x))
