import types
from functools import partial

from detectors.methods.ood.igeood import IgeoodLogits
from torch import nn

from .dice import Dice
from .energy import energy
from .knn_euclides import KnnEuclides
from .mahalanobis import Mahalanobis
from .msp import msp
from .odin import odin
from .react import ReAct


ood_detector_registry = {
    "msp": msp,
    "odin": odin,
    "energy": energy,
    "mahalanobis": Mahalanobis,
    "react": ReAct,
    "dice": Dice,
    "knn_euclides": KnnEuclides,
    "igeood_logits": IgeoodLogits,
}


def register_ood_detector(name: str):
    def decorator(f):
        ood_detector_registry[name] = f
        return f

    return decorator


def create_ood_detector(detector_name: str, model: nn.Module, **kwargs):
    if detector_name not in ood_detector_registry:
        raise ValueError(f"Unknown OOD detector: {detector_name}")
    if not isinstance(ood_detector_registry[detector_name], types.FunctionType):
        return ood_detector_registry[detector_name](model, **kwargs)
    return partial(ood_detector_registry[detector_name], model=model, **kwargs)


if __name__ == "__main__":
    import torch

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3)
            self.fc = nn.Linear(2700, 10)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = Model()
    detector = create_ood_detector("msp", model)
    x = torch.randn(1, 3, 32, 32)
    print(detector(x))
    assert detector(x).shape == (1,)
