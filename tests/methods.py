import pytest
import torch
from torch import Tensor

from detectors import create_detector


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.drop = torch.nn.Dropout()
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(6, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.drop(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


TEST_MODEL = Model()
TEST_MODEL.eval()
N = 100
X = torch.randn(N, 3, 32, 32)
Y = torch.randint(0, 2, (N,))
print(list(TEST_MODEL._modules.keys())[-2])


def test_msp():
    detector = create_detector("msp", model=TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)

    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)


def test_odin():
    hp = {"temperature": 1.0, "eps": 0.0}
    detector = create_detector("odin", model=TEST_MODEL, **hp)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)
    assert detector.keywords["temperature"] == hp["temperature"]
    assert detector.keywords["eps"] == hp["eps"]

    hp = {"temperature": 1000.0, "eps": 0.1}
    detector = create_detector("odin", model=TEST_MODEL, **hp)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert detector.keywords["temperature"] == hp["temperature"]
    assert detector.keywords["eps"] == hp["eps"]


@pytest.mark.parametrize("method_name", ["random", "always_one", "always_zero"])
def test_naive_scores(method_name):
    detector = create_detector(method_name)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    assert scores.shape == (N,)
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)


def test_energy():
    hp = {"temperature": 1.0}
    detector = create_detector("energy", model=TEST_MODEL, **hp)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)
    assert detector.keywords["temperature"] == hp["temperature"]

    hp = {"temperature": 1000.0}
    detector = create_detector("energy", model=TEST_MODEL, **hp)
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert detector.keywords["temperature"] == hp["temperature"]


def test_mahalanobis():
    detector = create_detector("mahalanobis", model=TEST_MODEL, features_nodes=["conv1"])
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


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


def test_dice():
    detector = create_detector("dice", model=TEST_MODEL, last_layer_name="linear", p=0.5)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_knn_euclides():
    detector = create_detector("knn_euclides", model=TEST_MODEL, features_nodes=["conv1"])
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_igeood_logits():
    detector = create_detector("igeood_logits", model=TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_max_logits():
    detector = create_detector("max_logits", model=TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_mc_dropout():
    detector = create_detector("mc_dropout", model=TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_kl_matching():
    detector = create_detector("kl_matching", model=TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_gradnorm():
    detector = create_detector("gradnorm", model=TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_maxcosine():
    detector = create_detector("maxcosine", model=TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_gmm():
    detector = create_detector("gmm", model=TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_relative_mahalanobis():
    detector = create_detector("relative_mahalanobis", model=TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)
