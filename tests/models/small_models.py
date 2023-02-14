import numpy as np
import torch
import detectors  # mandatory to register models in timm
import timm
from timm.data import resolve_data_config, create_transform
from PIL import Image


def test_cifar10_models():
    cifar10_models = timm.list_models("*cifar10*")
    for cf10_model in cifar10_models:
        model = timm.create_model(cf10_model, pretrained=False)
        input_size = model.default_cfg.input_size
        num_classes = model.default_cfg.num_classes
        assert num_classes == 10, f"num_classes should be 10, but got {num_classes}"
        x = torch.randn(1, *input_size)
        y = model(x)
        assert y.shape == (1, num_classes)

        # generate random image
        img = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        data_config = resolve_data_config(model.default_cfg)
        transform = create_transform(**data_config)
        x = transform(img)  # type: ignore
        assert x.shape == (3, 32, 32)  # type: ignore
        x = x.unsqueeze(0)  # type: ignore
        y = model(x)
        assert y.shape == (1, num_classes)


if __name__ == "__main__":
    test_cifar10_models()
