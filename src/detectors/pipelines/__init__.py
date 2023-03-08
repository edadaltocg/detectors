"""Pipeline module."""
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

    Returns:
        [Pipeline]: A suitable pipeline for the task.

    Examples:
    ```python
    >>> pipe = pipeline("ood_cifar10")
    ```
    """

    return pipeline_registry[task](**kwargs)


from .covariate_drift import *
from .ood import *
