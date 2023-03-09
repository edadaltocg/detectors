import pytest
import torchvision.transforms as transforms

import detectors


@pytest.mark.parametrize(
    "pipeline_name",
    [
        "ood_cifar10_benchmark",
        "ood_cifar100_benchmark",
        "ood_imagenet_benchmark",
        "ood_mnist_benchmark",
    ],
)
def test_ood_pipeline_benchmark(pipeline_name):
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
    detector = detectors.create_detector("random")
    pipeline = detectors.create_pipeline(pipeline_name, transform=transform, batch_size=512)
    pipeline = pipeline.run(detector)
