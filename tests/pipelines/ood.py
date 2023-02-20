import torchvision.transforms as transforms

import detectors


def test_pipeline_ood_cifar10():
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
    batch_size = 512
    pipeline = detectors.create_pipeline("ood_cifar10", transform=transform, batch_size=batch_size)
    detector = detectors.create_ood_detector("random")
    pipeline = pipeline.benchmark(detector)


def test_pipeline_ood_cifar100():
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
    batch_size = 512
    pipeline = detectors.create_pipeline("ood_cifar100", transform=transform, batch_size=batch_size)
    detector = detectors.create_ood_detector("random")
    pipeline = pipeline.benchmark(detector)


def test_pipeline_ood_imagenet():
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
    batch_size = 512
    pipeline = detectors.create_pipeline("ood_imagenet", transform=transform, batch_size=batch_size)
    detector = detectors.create_ood_detector("random")
    pipeline = pipeline.benchmark(detector)
