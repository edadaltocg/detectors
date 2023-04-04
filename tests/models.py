import numpy as np
import pytest
import timm
from PIL import Image
from timm.data import create_transform, resolve_data_config

import detectors  # mandatory to register models in timm
from detectors.models.densenet import default_cfgs as densenet_default_cfgs
from detectors.models.resnet import default_cfgs as resnet_default_cfgs
from detectors.models.vgg import default_cfgs as vgg_default_cfgs
from detectors.models.vit import default_cfgs as vit_default_cfgs

all_models = set(timm.list_models("*cifar10") + timm.list_models("*cifar100") + timm.list_models("*svhn"))
img = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
img = Image.fromarray(img)


@pytest.mark.parametrize("model_name", timm.list_models("*cifar10"))
def test_cifar10_architectures(model_name):
    model = timm.create_model(model_name, pretrained=False)
    model.default_cfg.input_size
    num_classes = model.default_cfg.num_classes
    data_config = resolve_data_config(model.default_cfg)
    transform = create_transform(**data_config)
    x = transform(img)  # type: ignore
    x = x.unsqueeze(0)  # type: ignore
    y = model(x)
    assert num_classes == 10
    assert y.shape == (1, num_classes)


@pytest.mark.parametrize("model_name", timm.list_models("*cifar100"))
def test_cifar100_architectures(model_name):
    model = timm.create_model(model_name, pretrained=False)
    model.default_cfg.input_size
    num_classes = model.default_cfg.num_classes
    data_config = resolve_data_config(model.default_cfg)
    transform = create_transform(**data_config)
    x = transform(img)  # type: ignore
    x = x.unsqueeze(0)  # type: ignore
    y = model(x)
    assert num_classes == 100
    assert y.shape == (1, num_classes)


@pytest.mark.parametrize("model_name", timm.list_models("*svhn"))
def test_svhn_architectures(model_name):
    model = timm.create_model(model_name, pretrained=False)
    model.default_cfg.input_size
    num_classes = model.default_cfg.num_classes
    data_config = resolve_data_config(model.default_cfg)
    transform = create_transform(**data_config)
    x = transform(img)  # type: ignore
    x = x.unsqueeze(0)  # type: ignore
    y = model(x)
    assert num_classes == 10
    assert y.shape == (1, num_classes)


@pytest.mark.parametrize("model_name", all_models)
def test_pretrained_model(model_name):
    model = timm.create_model(model_name, pretrained=True)
    model2 = timm.create_model(f"edadaltocg/{model_name}", pretrained=True)
    model.eval()
    model.default_cfg.input_size
    num_classes = model.default_cfg.num_classes
    # generate random image
    img = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img)
    data_config = resolve_data_config(model.default_cfg)
    transform = create_transform(**data_config)
    x = transform(img)  # type: ignore
    x = x.unsqueeze(0)  # type: ignore
    y = model(x)

    assert num_classes == num_classes
    assert y.shape == (1, num_classes)
