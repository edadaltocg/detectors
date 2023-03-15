"""Detection methods."""
import logging
import types
from functools import partial

from torch import Tensor
from detectors.methods.gradnorm import gradnorm

from detectors.methods.vim import ViM

from .dice import Dice
from .energy import energy
from .gmm import GMM
from .igeood import IgeoodLogits
from .kl_matching import KLMatching
from .knn_euclides import KnnEuclides
from .mahalanobis import Mahalanobis
from .max_logits import max_logits
from .mcdropout import mc_dropout
from .msp import msp
from .naive import always_one, always_zero, random_score
from .odin import odin
from .projection import Projection
from .react import ReAct
from .react_projection import ReActProjection

_logger = logging.getLogger(__name__)

detectors_registry = {
    "random": random_score,
    "always_one": always_one,
    "always_zero": always_zero,
    "msp": msp,
    "odin": odin,
    "max_logits": max_logits,
    "mc_dropout": mc_dropout,
    "energy": energy,
    "godin": ...,
    "mahalanobis": Mahalanobis,
    "react": ReAct,
    "dice": Dice,
    "knn_euclides": KnnEuclides,
    "igeood_logits": IgeoodLogits,
    "igeood_features": ...,
    "projection": Projection,
    "react_projection": ReActProjection,
    "bats": ...,
    "gram": ...,
    "rankfeat": ...,
    "vim": ViM,
    "kl_matching": KLMatching,
    "gradnorm": gradnorm,
    "gmm": GMM,
}


class Detector:
    """Detector wrapper."""

    def __init__(self, detector, **kwargs):
        self.detector = detector
        if hasattr(self.detector, "model"):
            self.model = self.detector.model
        elif hasattr(self.detector, "keywords") and "model" in self.detector.keywords:
            self.model = self.detector.keywords["model"]
        else:
            self.model = None
        self.keywords = kwargs

    def start(self, *args, **kwargs):
        if not hasattr(self.detector, "start"):
            _logger.warning("Detector does not have a start method.")
            return
        self.detector.start()

    def update(self, x: Tensor, y: Tensor):
        if not hasattr(self.detector, "update"):
            _logger.warning("Detector does not have an update method.")
            return
        self.detector.update(x, y)

    def end(self, *args, **kwargs):
        if not hasattr(self.detector, "end"):
            _logger.warning("Detector does not have an end method.")
            return
        self.detector.end()

    def fit(self, dataloader):
        self.start()
        for x, y in dataloader:
            self.update(x, y)
        self.end()
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
