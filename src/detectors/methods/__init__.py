"""
Detection methods.
"""
import logging
import types
from enum import Enum
from functools import partial
from typing import Any, Dict, List

from detectors.methods.templates import Detector, DetectorWrapper

from .dice import Dice
from .doctor import doctor
from .energy import energy
from .entropy import entropy
from .gmm import GMM
from .gradnorm import gradnorm
from .igeood_logits import IgeoodLogits
from .kl_matching import KLMatching
from .knn_euclides import KnnEuclides
from .mahalanobis import Mahalanobis
from .max_logits import max_logits
from .maxcosine import MaxCosineSimilarity
from .mcdropout import mcdropout
from .msp import msp
from .naive import always_one, always_zero, random_score
from .odin import odin
from .projection import Projection
from .react import ReAct
from .react_projection import ReActProjection
from .relative_mahalanobis import RelativeMahalanobis
from .ssd import SSD
from .vim import ViM

_logger = logging.getLogger(__name__)

detectors_registry = {
    # naive detectors
    "random": random_score,
    "always_one": always_one,
    "always_zero": always_zero,
    # hyperparameter free detectors
    "msp": msp,
    "max_logits": max_logits,
    "kl_matching": KLMatching,
    "vim": ViM,
    "mcdropout": mcdropout,
    "maxcosine": MaxCosineSimilarity,
    "entropy": entropy,
    # hyperparameter detectors
    "odin": odin,
    "doctor": doctor,
    "energy": energy,
    "dice": Dice,
    "react": ReAct,
    "igeood_logits": IgeoodLogits,
    "gradnorm": gradnorm,
    "knn_euclides": KnnEuclides,
    # Features based detectors
    "mahalanobis": Mahalanobis,
    "gmm": GMM,
    "relative_mahalanobis": RelativeMahalanobis,
    "projection": Projection,
    "react_projection": ReActProjection,
    # Special training detectors
    "ssd": SSD,
    # Not implemented
    "igeood_features": None,
    "bats": None,
    "gram": None,
    "openmax": None,
    "rankfeat": None,
    "godin": None,
}


def register_detector(name: str):
    """Decorator to register a new detector.

    Args:
        name (string): Name of the detector.

    Example::

        @register_detector("my_detector")
        class MyDetector(Detector):
            ...

        detector = create_detector("my_detector")

        @register_detector("my_detector")
        def my_detector(model, **kwargs):
            ...

        detector = create_detector("my_detector")
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
                `projection`, `react_projection`, `gradnorm`, `maxcosine`, `mcdropout`, `max_logits`, `kl_matching`,
                `gmm`, `relative_mahalanobis`, `doctor`, `always_one`, `always_zero`, `random_score`, `vim`,
                `entropy`, `ssd`
        **kwargs: Additional arguments for the detector.

    Returns:
        Detector: the corresponding detector.
    """
    model = kwargs.pop("model", None)
    if detector_name not in detectors_registry:
        raise ValueError(f"Unknown detector: {detector_name}")
    if not isinstance(detectors_registry[detector_name], types.FunctionType):
        return DetectorWrapper(detectors_registry[detector_name](model=model, **kwargs), **kwargs)
    return DetectorWrapper(partial(detectors_registry[detector_name], model=model, **kwargs), **kwargs)


def list_detectors() -> List[str]:
    """List available detectors.

    Returns:
        List[str]: List of available detectors.
    """
    return list(k for k in detectors_registry.keys() if detectors_registry[k] is not None)


def create_hyperparameters(detector_name: str) -> Dict[str, Any]:
    """Create hyperparameters for the detector.

    Args:
        detector_name (string): Name of the detector.

    Returns:
        Dict[str, Any]: Hyperparameters for the detector.
    """
    import importlib

    try:
        module = importlib.import_module(f"detectors.methods.{detector_name}")
        hyperparameters = module.HYPERPARAMETERS
    except ModuleNotFoundError:
        raise ValueError(f"Unknown detector: {detector_name}")
    except AttributeError:
        hyperparameters = {}
    return hyperparameters


MethodsRegistry = Enum("MethodsRegistry", dict(zip(list_detectors(), list_detectors())))
