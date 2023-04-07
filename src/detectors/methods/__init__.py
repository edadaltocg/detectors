"""Detection methods."""
import logging
import types
from enum import Enum
from functools import partial
from typing import Any, Dict

from detectors.methods.templates import Detector, DetectorWrapper

from .dice import Dice
from .doctor import doctor
from .energy import energy
from .gmm import GMM
from .gradnorm import gradnorm
from .igeood_logits import IgeoodLogits
from .kl_matching import KLMatching
from .knn_euclides import KnnEuclides
from .logit_norm import logit_norm
from .mahalanobis import Mahalanobis
from .max_logits import max_logits
from .maxcosine import MaxCosineSimilarity
from .mcdropout import mc_dropout
from .msp import msp
from .naive import always_one, always_zero, random_score
from .odin import odin
from .projection import Projection
from .react import ReAct
from .react_projection import ReActProjection
from .relative_mahalanobis import RelativeMahalanobis
from .vim import ViM

_logger = logging.getLogger(__name__)

detectors_registry = {
    "random": random_score,
    "always_one": always_one,
    "always_zero": always_zero,
    "msp": msp,
    "odin": odin,
    "doctor": doctor,
    "max_logits": max_logits,
    "mcdropout": mc_dropout,
    "energy": energy,
    "mahalanobis": Mahalanobis,
    "relative_mahalanobis": RelativeMahalanobis,
    "react": ReAct,
    "dice": Dice,
    "knn_euclides": KnnEuclides,
    "igeood_logits": IgeoodLogits,
    "igeood_features": ...,
    "projection": Projection,
    "react_projection": ReActProjection,
    "godin": ...,
    "bats": ...,
    "gram": ...,
    "openmax": ...,
    "rankfeat": ...,
    "vim": ViM,
    "kl_matching": KLMatching,
    "gradnorm": gradnorm,
    "gmm": GMM,
    "maxcosine": MaxCosineSimilarity,
    "logit_norm": ...,
}


def register_detector(name: str):
    """Decorator to register a new detector.

    Args:
        name (string): Name of the detector.
    """

    def decorator(f):
        detectors_registry[name] = f
        return f

    return decorator


def create_detector(detector_name: str, **kwargs) -> Detector:
    """Create detector factory.

    Args:
        detector_name (string): Name of the detector.
            Already implemented:
                `random`, `msp`, `odin`, `energy`, `mahalanobis`, `react`, `dice`, `knn_euclides`, `igeood_logits`,
                `projection`, `react_projection`
        **kwargs: Additional arguments for the detector.

    Returns:
        Detector
    """
    model = kwargs.pop("model", None)
    if detector_name not in detectors_registry:
        raise ValueError(f"Unknown detector: {detector_name}")
    if not isinstance(detectors_registry[detector_name], types.FunctionType):
        return DetectorWrapper(detectors_registry[detector_name](model=model, **kwargs), **kwargs)
    return DetectorWrapper(partial(detectors_registry[detector_name], model=model, **kwargs), **kwargs)


def list_detectors():
    """List available detectors."""
    return list(detectors_registry.keys())


def create_hyperparameters(detector_name: str) -> Dict[str, Any]:
    """Create hyperparameters for the detector.

    Args:
        detector_name (string): Name of the detector.

    Returns:
        dict: Hyperparameters for the detector.
    """
    import importlib

    try:
        module = importlib.import_module(f"detectors.methods.{detector_name}")
        hyperparameters = module.HYPERPARAMETERS
    except ModuleNotFoundError:
        hyperparameters = {}
    return hyperparameters


MethodsRegistry = Enum("MethodsRegistry", dict(zip(list_detectors(), list_detectors())))
