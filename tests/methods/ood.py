import logging

import torch
from torch import Tensor

import detectors
from detectors import create_ood_detector


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.drop = torch.nn.Dropout()
        self.linear = torch.nn.Linear(6 * 28 * 28, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.drop(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


TEST_MODEL = Model()
TEST_MODEL.eval()
N = 100
X = torch.randn(N, 3, 32, 32)
Y = torch.randint(0, 2, (N,))


def test_msp():
    detector = create_ood_detector("msp", TEST_MODEL)
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
    detector = create_ood_detector("odin", TEST_MODEL, **hp)
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
    detector = create_ood_detector("odin", TEST_MODEL, **hp)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert detector.keywords["temperature"] == hp["temperature"]
    assert detector.keywords["eps"] == hp["eps"]


def test_random_score():
    detector = create_ood_detector("random", TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)


def test_energy():
    hp = {"temperature": 1.0}
    detector = create_ood_detector("energy", TEST_MODEL, **hp)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)
    assert detector.keywords["temperature"] == hp["temperature"]

    hp = {"temperature": 1000.0}
    detector = create_ood_detector("energy", TEST_MODEL, **hp)
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert detector.keywords["temperature"] == hp["temperature"]


def test_mahalanobis():
    detector = create_ood_detector("mahalanobis", TEST_MODEL, features_nodes=["conv1"])
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_react():
    detector = create_ood_detector("react", TEST_MODEL, features_nodes=["conv1"])
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)
    assert not torch.allclose(scores, torch.logsumexp(TEST_MODEL(X), dim=-1))


def test_dice():
    detector = create_ood_detector("dice", TEST_MODEL, last_layer_name="linear", p=0.5)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_knn_euclides():
    detector = create_ood_detector("knn_euclides", TEST_MODEL, features_nodes=["conv1"])
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_igeood_logits():
    detector = create_ood_detector("igeood_logits", TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_max_logits():
    detector = create_ood_detector("max_logits", TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_mc_dropout():
    detector = create_ood_detector("mc_dropout", TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


def test_kl_matching():
    detector = create_ood_detector("kl_matching", TEST_MODEL)
    detector.start()
    detector.update(X, Y)
    detector.end()
    scores = detector(X)
    scores_std = scores.std()
    assert scores_std > 0.0
    assert scores.shape == (N,)


if __name__ == "__main__":
    test_kl_matching()
