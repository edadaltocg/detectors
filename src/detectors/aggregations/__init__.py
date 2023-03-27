import types
from abc import ABC, abstractmethod
from functools import partial
from torch import Tensor
from detectors.aggregations.anomaly import IFAggregation, LOFAggregation
import logging
from detectors.aggregations.basics import (
    avg_topk_aggregation,
    depth_weighted_sum,
    layer_idx,
    max_aggregation,
    mean_aggregation,
    min_aggregation,
    topk_aggregation,
    median_aggregation,
)
from detectors.aggregations.cosine import CosineAggregation
from detectors.aggregations.innerprod import InnerProductAggregation
from detectors.aggregations.mahalanobis import MahalanobisAggregation


_logger = logging.getLogger(__name__)
aggregations_registry = {
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
    "cosine": CosineAggregation,
}


class Aggregation:
    """Aggregation wrapper class."""

    def __init__(self, aggregation_method, *args, **kwargs) -> None:
        self.aggregation_method = aggregation_method

    def fit(self, stack: Tensor, *args, **kwargs):
        if not hasattr(self.aggregation_method, "fit"):
            _logger.warning("Aggregation method does not have a `fit` method.")
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
