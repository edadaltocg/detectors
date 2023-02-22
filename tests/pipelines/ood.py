import pytest
import torchvision.transforms as transforms

import detectors


@pytest.mark.parametrize("pipeline_name", ["ood_cifar10", "ood_cifar100", "ood_imagenet"])
def test_ood_pipeline(pipeline_name):
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
    batch_size = 512
    pipeline = detectors.create_pipeline(pipeline_name, transform=transform, batch_size=batch_size)
    detector = detectors.create_ood_detector("random")
    pipeline = pipeline.run(detector)
