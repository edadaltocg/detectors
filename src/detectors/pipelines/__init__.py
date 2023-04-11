"""
Pipeline module.
"""
from enum import Enum
from typing import Any, List, Tuple

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


def create_pipeline(name: str, **kwargs) -> Pipeline:
    """
    Utility factory method to build a Pipeline.

    Args:
        name (str, optional):
            The name defining which pipeline will be returned. Currently accepted pipeline names are:
                `ood_benchmark_cifar10`, `ood_benchmark_cifar100`, `ood_benchmark_imagenet`, `ood_mnist_benchmark`,
                `ood_validation_cifar10`, `ood_validation_cifar100`, `ood_validation_imagenet`, `ood_validation_mnist`,
                `ood_validation_noise_cifar10`, `ood_validation_noise_cifar100`, `ood_validation_noise_imagenet`,

        **kwargs:
            Additional arguments to pass to the pipeline.

    Returns:
        Pipeline: A suitable pipeline for the task.

    Example::

        >>> import detectors
        >>> pipe = detectors.create_pipeline("ood_benchmark_cifar10")
        >>> pipe.run(detector)
    """

    return pipeline_registry[name](**kwargs)


def list_pipelines() -> List[str]:
    """
    List all available pipelines.

    Returns:
        List[str]: A list of available pipelines.
    """
    return list(pipeline_registry.keys())


def list_pipeline_args(name: str) -> List[Tuple[str, Any]]:
    """
    List all available arguments for a given pipeline.

    Args:
        name (str): The name of the pipeline.

    Returns:
        list: A list of available arguments and default values for the pipeline.
    """
    import inspect

    signature = inspect.signature(pipeline_registry[name]).parameters
    return [(name, parameter.default) for name, parameter in signature.items()]


from .covariate_drift import *
from .ood import *

PipelinesRegistry = Enum("PipelinesRegistry", dict(zip(list_pipelines(), list_pipelines())))
if __name__ == "__main__":
    print(list_pipeline_args("ood_benchmark_cifar10"))
