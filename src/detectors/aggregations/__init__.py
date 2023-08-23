import logging
import types
from functools import partial

from torch import Tensor

from .anomaly import IFAggregation, LOFAggregation
from .basics import (
    avg_topk_aggregation,
    depth_weighted_sum,
    layer_idx,
    max_aggregation,
    mean_aggregation,
    median_aggregation,
    min_aggregation,
    none_aggregation,
    topk_aggregation,
)
from .blahut_arimoto import BlahutArimotoAggregation
from .cosine import CosineAggregation
from .innerprod import (
    InnerProductAggregation,
    InnerProductIntegralAggregation,
    InnerProductMeanAggregation,
    InnerProductMinAggregation,
)
from .mahalanobis import MahalanobisAggregation
from .power import PowerAggregation
from .quantile import QuantileAggregation

_logger = logging.getLogger(__name__)

aggregations_registry = {
    "none": none_aggregation,
    "mean": mean_aggregation,
    "max": max_aggregation,
    "min": min_aggregation,
    "median": median_aggregation,
    "dws": depth_weighted_sum,
    "avg_topk": avg_topk_aggregation,
    "topk": topk_aggregation,
    "lof": LOFAggregation,
    "if": IFAggregation,
    "layer_idx": layer_idx,
    "mahalanobis": MahalanobisAggregation,
    "innerprod": InnerProductAggregation,
    "innerprod_mean": InnerProductMeanAggregation,
    "innerprod_min": InnerProductMinAggregation,
    "innerprod_integral": InnerProductIntegralAggregation,
    "cosine": CosineAggregation,
    "quantile": QuantileAggregation,
    "power": PowerAggregation,
    "blahut_arimoto": BlahutArimotoAggregation,
}


class Aggregation:
    """Aggregation wrapper class."""

    def __init__(self, aggregation_method, *args, **kwargs) -> None:
        self.aggregation_method = aggregation_method

    def fit(self, stack: Tensor, *args, **kwargs):
        if not hasattr(self.aggregation_method, "fit"):
            _logger.debug("Aggregation method does not have a `fit` method.")
            return
        self.aggregation_method.fit(stack)

    def __call__(self, stack: Tensor, *args, **kwargs):
        return self.aggregation_method(stack, *args, **kwargs)


def register_aggregation(name: str):
    """Decorator to register a new aggregation method."""

    def decorator(f):
        aggregations_registry[name] = f
        return f

    return decorator


def create_aggregation(aggregation_name: str, **kwargs) -> Aggregation:
    if aggregation_name not in aggregations_registry:
        raise ValueError(f"Unknown aggregation method: {aggregation_name}")
    if not isinstance(aggregations_registry[aggregation_name], types.FunctionType):
        return Aggregation(aggregations_registry[aggregation_name](**kwargs))
    return Aggregation(partial(aggregations_registry[aggregation_name], **kwargs))


def list_aggregations() -> list:
    return list(aggregations_registry.keys())
