import logging
import types
from functools import partial

from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from torch import Tensor

from .basics import none_preprocessing

_logger = logging.getLogger(__name__)

preprocessings_registry = {
    "none": none_preprocessing,
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
    "quantile_uniform": QuantileTransformer,
    "normalizer_l2": Normalizer,
    "power": PowerTransformer,
}


class Preprocessing:
    """Aggregation wrapper class."""

    def __init__(self, preprocessing_method, **kwargs) -> None:
        self.method = preprocessing_method

    def fit(self, x: Tensor, *args, **kwargs):
        if not hasattr(self.method, "fit"):
            _logger.debug("Aggregation method does not have a `fit` method.")
            return
        x = x.detach().cpu().numpy()
        self.method.fit(x)

    def transform(self, x: Tensor, *args, **kwargs):
        x = x.detach().cpu().numpy()
        try:
            return self.method.transform(x, *args, **kwargs)
        except Exception as e:
            return self.method(x, *args, **kwargs)

    def __call__(self, x: Tensor, *args, **kwargs):
        return self.transform(x, *args, **kwargs)


def register_preprocessing(name: str):
    """Decorator to register a new preprocessing method."""

    def decorator(f):
        preprocessings_registry[name] = f
        return f

    return decorator


def create_preprocessing(preprocessing_name: str, **kwargs) -> Preprocessing:
    if preprocessing_name not in preprocessings_registry:
        raise ValueError(f"Unknown aggregation method: {preprocessing_name}")
    if not isinstance(preprocessings_registry[preprocessing_name], types.FunctionType):
        return Preprocessing(preprocessings_registry[preprocessing_name](**kwargs))
    return Preprocessing(partial(preprocessings_registry[preprocessing_name], **kwargs))


def list_preprocessings() -> list:
    return list(preprocessings_registry.keys())
