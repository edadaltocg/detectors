import torch
import torch.nn as nn
import torchvision


def DenseNet121Small(num_classes=10):
    model = torchvision.models.densenet._densenet(
        growth_rate=12, block_config=(6, 12, 24, 16), num_init_features=24, weights=None, progress=False
    )
    model.features.conv0 = nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False)
    del model.features.norm0
    del model.features.relu0
    del model.features.pool0

    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


if __name__ == "__main__":
    model = DenseNet121Small()
    print(model)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
