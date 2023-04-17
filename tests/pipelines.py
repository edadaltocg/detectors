import pytest
import torchvision.transforms as transforms

import detectors


@pytest.mark.parametrize(
    "pipeline_name",
    [
        "ood_benchmark_cifar10",
        "ood_benchmark_cifar100",
        "ood_benchmark_imagenet",
        "ood_mnist_benchmark",
    ],
)
def test_ood_pipeline_benchmark(pipeline_name):
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
    detector = detectors.create_detector("random")
    pipeline = detectors.create_pipeline(pipeline_name, transform=transform, batch_size=512)
    pipeline = pipeline.run(detector)


@pytest.mark.parametrize(
    "pipeline_name",
    [
        "osr_cifar10",
        "osr_cifar100",
    ],
)
def test_osr_pipeline(pipeline_name):
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
    detector = detectors.create_detector("random")
    pipeline = detectors.create_pipeline(pipeline_name, transform=transform, batch_size=512)
    pipeline = pipeline.run(detector)


@pytest.mark.parametrize(
    "pipeline_name",
    [
        "one_corruption_covariate_drift_cifar10",
        "one_corruption_covariate_drift_cifar100",
        "one_corruption_covariate_drift_imagenet",
    ],
)
def test_covariate_drift_pipeline(pipeline_name):
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
    detector = detectors.create_detector("random")
    pipeline = detectors.create_pipeline(
        pipeline_name, corruption="brightness", intensities=[1, 3, 5], transform=transform, batch_size=512
    )
    pipeline = pipeline.run(detector)


@pytest.mark.parametrize(
    "pipeline_name",
    [
        "ood_cifar10_noise_validation",
        "ood_cifar100_noise_validation",
        # "ood_imagenet_noise_validation",
    ],
)
def test_ood_pipeline_noise_validation(pipeline_name):
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
    detector = detectors.create_detector("random")
    pipeline = detectors.create_pipeline(pipeline_name, transform=transform, batch_size=512)
    hyperparameters = {}
    pipeline = pipeline.run(detector, hyperparameters, n_trials=2)


@pytest.mark.parametrize(
    "pipeline_name",
    [
        "ood_validation_cifar10",
        "ood_validation_cifar100",
        "ood_validation_imagenet",
    ],
)
def test_ood_validation_pipeline(pipeline_name):
    method_name = "mahalanobis"
    model = detectors.create_model("densenet121_cifar10", pretrained=True)
    transform = detectors.create_transform(model)
    method = detectors.create_detector(method_name, model=model)
    pipeline = detectors.create_pipeline(pipeline_name, transform=transform, batch_size=512, limit_run=0.01)
    hyperparameters = detectors.methods.create_hyperparameters(method_name)
    print(hyperparameters)
    results = pipeline.run(method, hyperparameters=hyperparameters, n_trials=3)
    print(pipeline.report(results))
