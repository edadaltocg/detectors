"""
Pipeline module.
"""
from enum import Enum
from typing import Any, List, Optional, Tuple

from detectors.pipelines.base import Pipeline

pipeline_registry = {}


def register_pipeline(name: str, dataset_name: Optional[str] = None):
    """
    Decorator to register a new pipeline in the registry.

    Args:
        name (str): The name of the pipeline to register.
    """

    def decorator(f):
        if dataset_name is None:
            pipeline_registry[name] = f
        else:
            pipeline_registry[name + "_" + dataset_name] = f
        return f

    return decorator


## str(detectors.create_pipeline()).replace("'", "`")
def create_pipeline(name: str, dataset_name: Optional[str] = None, **kwargs) -> Pipeline:
    """
    Utility factory method to build a Pipeline.

    Args:
        name (str, optional):
            The name defining which pipeline will be returned. Currently accepted pipeline names are:
                `ood_benchmark`, `ood_validation`, `ood_validation_noise`, `covariate_drift`, `osr`
            Complete list:
                `covariate_drift_cifar10`, `covariate_drift_cifar100`, `covariate_drift_imagenet`,
                `drift_benchmark_imagenet_r`, `drift_benchmark_imagenet_lt`, `drift_benchmark_openimage_o`,
                `ood_benchmark_cifar10`, `ood_benchmark_cifar100`, `ood_benchmark_imagenet`,
                `ood_benchmark_imagenet_reduced`, `ood_benchmark_imagenet_near`, `ood_benchmark_imagenet_all_2`,
                `ood_benchmark_imagenet_far`, `ood_mnist_benchmark`, `ood_validation_cifar10`,
                `ood_validation_noise_cifar10`, `ood_validation_cifar100`, `ood_validation_noise_cifar100`,
                `ood_validation_imagenet`, `sc_benchmark_cifar10`, `sc_benchmark_cifar100`,
                `sc_benchmark_imagenet`, `scod_benchmark_cifar10`, `scod_benchmark_cifar100`
        **kwargs:
            Additional arguments to pass to the pipeline.

    Returns:
        Pipeline: A suitable pipeline for the task.

    Example::

        >>> import detectors
        >>> pipe = detectors.create_pipeline("ood_benchmark", "cifar10")
        >>> pipe.run(detector)
    """
    if dataset_name is None:
        return pipeline_registry[name](**kwargs)
    return pipeline_registry[name + "_" + dataset_name](**kwargs)


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
from .drift import *
from .ood import *
from .sc import *
from .scod import *

PipelinesRegistry = Enum("PipelinesRegistry", dict(zip(list_pipelines(), list_pipelines())))
