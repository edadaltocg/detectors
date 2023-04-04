"""Detection methods."""
import logging
import types
from enum import Enum
from functools import partial
from typing import Optional

from torch import Tensor

from .dice import Dice
from .doctor import doctor
from .energy import energy
from .gmm import GMM
from .gradnorm import gradnorm
from .igeood import IgeoodLogits
from .kl_matching import KLMatching
from .knn_euclides import KnnEuclides
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
    "mc_dropout": mc_dropout,
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
}


class Detector:
    """Detector interface."""

    def __init__(self, detector, **kwargs):
        self.detector = detector
        if hasattr(self.detector, "model"):
            self.model = self.detector.model
        elif hasattr(self.detector, "keywords") and "model" in self.detector.keywords:
            self.model = self.detector.keywords["model"]
        else:
            self.model = None
        self.keywords = kwargs

    def start(self, example: Optional[Tensor] = None, fit_length: Optional[int] = None, *args, **kwargs):
        if not hasattr(self.detector, "start"):
            _logger.warning("Detector does not have a start method.")
            return
        self.detector.start(example, fit_length, *args, **kwargs)

    def update(self, x: Tensor, y: Tensor, *args, **kwargs):
        if not hasattr(self.detector, "update"):
            _logger.warning("Detector does not have an update method.")
            return
        self.detector.update(x, y, *args, **kwargs)

    def end(self, *args, **kwargs):
        if not hasattr(self.detector, "end"):
            _logger.warning("Detector does not have an end method.")
            return
        self.detector.end(*args, **kwargs)

    def fit(self, dataloader, **kwargs):
        # get fit length # CHECK BUG
        fit_length = len(dataloader.dataset)
        # get example
        x, y = next(iter(dataloader))
        self.start(example=x, fit_length=fit_length, **kwargs)
        for x, y in dataloader:
            self.update(x, y, **kwargs)
        self.end(**kwargs)
        return self

    def __call__(self, x: Tensor) -> Tensor:
        """

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: scores for each input.
        """
        return self.detector(x)

    def set_params(self, **params):
        model = params.pop("model", None)
        self.keywords.update(params)
        if hasattr(self.detector, "keywords"):
            self.detector.keywords.update(**params)
        else:
            self.detector = self.detector.__class__(model=model, **self.keywords)
        return self

    # def set_params(self, **params):
    #     """Set the parameters of the detector."""
    #     raise NotImplementedError

    def save_params(self, path):
        """Save the parameters of the detector."""
        raise NotImplementedError

    def load_params(self, path):
        """Load the parameters of the detector."""
        raise NotImplementedError

    def __repr__(self):
        """Return the string representation of the detector."""
        return f"{self.__class__.__name__}()"


def register_detector(name: str):
    """Decorator to register a new detector."""

    def decorator(f):
        detectors_registry[name] = f
        return f

    return decorator


def create_detector(detector_name: str, **kwargs) -> Detector:
    """Create detector factory.

    Args:
        detector_name (string): Name of the detector.
            Already implemented: [`random`, `msp`, `odin`, `energy`, `mahalanobis`, `react`, `dice`, `knn_euclides`, `igeood_logits`, `projection`, `react_projection`]
        model (nn.Module): Model to be used for the OOD detector.
        **kwargs: Additional arguments for the detector.

    Returns:
        Detector.
    """
    model = kwargs.pop("model", None)
    if detector_name not in detectors_registry:
        raise ValueError(f"Unknown detector: {detector_name}")
    if not isinstance(detectors_registry[detector_name], types.FunctionType):
        return Detector(detectors_registry[detector_name](model=model, **kwargs), **kwargs)
    return Detector(partial(detectors_registry[detector_name], model=model, **kwargs), **kwargs)


def list_detectors():
    """List available detectors."""
    return list(detectors_registry.keys())


MethodsRegistry = Enum("MethodsRegistry", dict(zip(list_detectors(), list_detectors())))
