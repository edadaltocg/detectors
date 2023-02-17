import types
from torch import Tensor
from functools import partial

from torch import nn

from detectors.methods.ood.igeood import IgeoodLogits

from .dice import Dice
from .energy import energy
from .knn_euclides import KnnEuclides
from .mahalanobis import Mahalanobis
from .msp import msp
from .odin import odin
from .projection import Projection
from .random import random_score
from .react import ReAct
from .react_projection import ReActProjection
import logging

_logger = logging.getLogger(__name__)
ood_detector_registry = {
    "random": random_score,
    "msp": msp,
    "odin": odin,
    "godin": ...,
    "energy": energy,
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
    "vim": ...,
    "kl_logits": ...,
}


class OODDetector:
    def __init__(self, detector, model: nn.Module, **kwargs):
        self.detector = detector
        self.model = model
        self.keywords = kwargs

    def start(self, *args, **kwargs):
        if not hasattr(self.detector, "start"):
            _logger.warning(f"Detector {self.detector} does not have a start method")
            return
        self.detector.start()

    def update(self, x: Tensor, y: Tensor):
        if not hasattr(self.detector, "update"):
            _logger.warning(f"Detector {self.detector} does not have an update method")
            return
        self.detector.update(x, y)

    def end(self, *args, **kwargs):
        if not hasattr(self.detector, "end"):
            _logger.warning(f"Detector {self.detector} does not have an end method")
            return
        self.detector.end()

    def __call__(self, x: Tensor) -> Tensor:
        return self.detector(x)

    def set_params(self, **params):
        """Set the parameters of this detector.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        raise NotImplementedError
        return self


def register_ood_detector(name: str):
    def decorator(f):
        ood_detector_registry[name] = f
        return f

    return decorator


def create_ood_detector(detector_name: str, model: nn.Module, **kwargs) -> OODDetector:
    if detector_name not in ood_detector_registry:
        raise ValueError(f"Unknown OOD detector: {detector_name}")
    if not isinstance(ood_detector_registry[detector_name], types.FunctionType):
        return OODDetector(ood_detector_registry[detector_name](model, **kwargs), model, **kwargs)
    return OODDetector(partial(ood_detector_registry[detector_name], model=model, **kwargs), model, **kwargs)
