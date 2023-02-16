import numpy as np
import timm
from PIL import Image
from timm.data import create_transform, resolve_data_config

import detectors  # mandatory to register models in timm
from detectors.models.densenet import default_cfgs as densenet_default_cfgs
from detectors.models.resnet import default_cfgs as resnet_default_cfgs
from detectors.models.vgg import default_cfgs as vgg_default_cfgs
from detectors.models.vit import default_cfgs as vit_default_cfgs


def test_list_has_models():
    models = set(timm.list_models("*cifar10") + timm.list_models("*cifar100") + timm.list_models("*svhn"))
    all_models = set(
        list(densenet_default_cfgs.keys())
        + list(resnet_default_cfgs.keys())
        + list(vgg_default_cfgs.keys())
        + list(vit_default_cfgs.keys())
    )
    print(models, all_models)
    assert len(models) == len(all_models)


def test_cifar10_ce_models():
    cifar10_models = timm.list_models("*cifar10")
    assert len(cifar10_models) > 0
    for cf10_model in cifar10_models:
        model = timm.create_model(cf10_model, pretrained=False)
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
        assert num_classes == 10
        assert y.shape == (1, num_classes)


def test_cifar100_ce_models():
    cifar100_models = timm.list_models("*cifar100")
    assert len(cifar100_models) > 0
    for cf100_model in cifar100_models:
        model = timm.create_model(cf100_model, pretrained=False)
        model.default_cfg.input_size
        num_classes = model.default_cfg.num_classes
        # generate random image
        img = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        data_config = resolve_data_config(model.default_cfg)
        transform = create_transform(**data_config)
        x = transform(img)
        x = x.unsqueeze(0)
        y = model(x)
        assert num_classes == 100
        assert y.shape == (1, num_classes)


def test_svhn_ce_models():
    svhn_models = timm.list_models("*svhn")
    assert len(svhn_models) > 0
    for svhn_model in svhn_models:
        model = timm.create_model(svhn_model, pretrained=False)
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
        assert num_classes == 10
        assert y.shape == (1, num_classes)
