import pytest
import torch
from torch import Tensor

from detectors import create_detector, create_hyperparameters, create_model


torch.manual_seed(0)
TEST_MODEL = create_model("resnet18_cifar10")
TEST_MODEL.eval()
N = 128
X = torch.randn(N, 3, 32, 32)
Y = torch.randint(0, 2, (N,))


@pytest.mark.parametrize("method_name", ["random", "always_one", "always_zero"])
def test_naive_detectors(method_name):
    detector = create_detector(method_name)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    assert scores.shape == (N,)
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)


@pytest.mark.parametrize(
    "method_name",
    [
        "odin",
        "doctor",
        "energy",
        "dice",
        "react",
        "igeood_logits",
        "gradnorm",
        "knn_euclides",
    ],
)
def test_detectors_with_hyperparameters(method_name):
    detector = create_detector(method_name, model=TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)

    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)

    hyperparameters = create_hyperparameters(method_name)
    assert len(hyperparameters) > 0


@pytest.mark.parametrize(
    "method_name",
    [
        "msp",
        "max_logits",
        "kl_matching",
        "vim",
        "mcdropout",
        "maxcosine",
    ],
)
def test_hyperparameter_free_detectors(method_name):
    detector = create_detector(method_name, model=TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)

    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)
    hyperparameters = create_hyperparameters(method_name)
    assert len(hyperparameters) == 0


def test_react():
    detector = create_detector("react", model=TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)
    assert not torch.allclose(scores, torch.logsumexp(TEST_MODEL(X), dim=-1))
