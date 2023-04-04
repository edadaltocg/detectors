"""Pipeline module."""
from enum import Enum

from detectors.pipelines.base import Pipeline

pipeline_registry = {}


def register_pipeline(name: str):
    """
    Decorator to register a new pipeline in the registry.

    Args:
        name (str): The name of the pipeline to register.
    """

    def decorator(f):
        pipeline_registry[name] = f
        return f

    return decorator


def create_pipeline(task: str, **kwargs) -> Pipeline:
    """
    Utility factory method to build a [`Pipeline`].

    Args:
        task (str, optional):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"ood-cifar"`: will return a [`OODCIFARPipeline`].

        seed (int, optional):
            The seed to use for the pipeline.

    Returns:
        [Pipeline]: A suitable pipeline for the task.

    Examples:

        ```python
    >>> pipe = pipeline("ood_cifar10_benchmark")
        ```
    """

    return pipeline_registry[task](**kwargs)


def list_pipelines() -> list:
    """
    List all available pipelines.

    Returns:
        list: A list of available pipelines.
    """
    return list(pipeline_registry.keys())


from .covariate_drift import *
from .ood import *

PipelinesRegistry = Enum("PipelinesRegistry", dict(zip(list_pipelines(), list_pipelines())))
