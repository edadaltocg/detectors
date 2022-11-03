from functools import partial

from detectors.models import create_model
from .msp import msp
from .odin import odin

ood_detector_registry = {
    "msp": msp,
    "odin": odin,
}


def register_ood_detector(name: str):
    def decorator(f):
        ood_detector_registry[name] = f
        return f

    return decorator


def create_ood_detector(detector_name: str, model_name: str, **kwargs):
    if detector_name not in ood_detector_registry:
        raise ValueError(f"Unknown OOD detector: {detector_name}")
    weights = kwargs.pop("weights", False)
    method = partial(ood_detector_registry[detector_name], model=create_model(model_name, weights=weights), **kwargs)
    return method


if __name__ == "__main__":
    import torch

    detector = create_ood_detector("msp", "resnet34_cifar10")
    x = torch.randn(1, 3, 32, 32)
    print(detector(x))
