import logging
import types
from functools import partial
from typing import Optional

from torch import Tensor, nn

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
    "mls": ...,
}


class OODDetector:
    def __init__(self, detector, model: Optional[nn.Module] = None, **kwargs):
        self.detector = detector
        self.model = model
        self.keywords = kwargs

    def start(self, *args, **kwargs):
        if not hasattr(self.detector, "start"):
            _logger.warning("Detector does not have a start method")
            return
        self.detector.start()

    def update(self, x: Tensor, y: Tensor):
        if not hasattr(self.detector, "update"):
            _logger.warning("Detector does not have an update method")
            return
        self.detector.update(x, y)

    def end(self, *args, **kwargs):
        if not hasattr(self.detector, "end"):
            _logger.warning("Detector  does not have an end method")
            return
        self.detector.end()

    def __call__(self, x: Tensor) -> Tensor:
        return self.detector(x)

    def set_params(self, **params):
        """Set the parameters of this detector.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : detector instance
        """
        self.keywords.update(params)
        if hasattr(self.detector, "keywords"):
            self.detector.keywords.update(**params)
        else:
            self.detector = self.detector.__class__(model=self.model, **self.keywords)
        return self


def register_ood_detector(name: str):
    def decorator(f):
        ood_detector_registry[name] = f
        return f

    return decorator


def create_ood_detector(detector_name: str, model: Optional[nn.Module] = None, **kwargs) -> OODDetector:
    if detector_name not in ood_detector_registry:
        raise ValueError(f"Unknown OOD detector: {detector_name}")
    if not isinstance(ood_detector_registry[detector_name], types.FunctionType):
        return OODDetector(ood_detector_registry[detector_name](model, **kwargs), model, **kwargs)
    return OODDetector(partial(ood_detector_registry[detector_name], model=model, **kwargs), model, **kwargs)
