from typing import Optional, Union

import torch

from detectors.pipelines.base import Pipeline

pipeline_registry = {}


def register_pipeline(name: str):
    """
    Registers a new pipeline in the registry.

    Args:
        name (str): The name of the pipeline to register.
    """

    def decorator(f):
        pipeline_registry[name] = f
        return f

    return decorator


def pipeline(
    task: str,
    device: Optional[Union[int, str, "torch.device"]] = None,
    **kwargs,
) -> Pipeline:
    """
    Utility factory method to build a [`Pipeline`].

    Pipelines are made of:

        - A [tokenizer](tokenizer) in charge of mapping raw textual input to token.
        - A [model](model) to make predictions from the inputs.
        - Some (optional) post processing for enhancing model's output.

    Args:
        task (str, optional):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"ood-cifar"`: will return a [`OODCIFARPipeline`].

        method (Optional, optional):
            _description_. Defaults to None.


        device (Optional[Union[int, str, &quot;torch.device&quot;]], optional):
            Defines the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank like `1`) on which this
            pipeline will be allocated.

    Returns:
        [Pipeline]: A suitable pipeline for the task.

    Examples:

    ```python
    >>> x = torch.rand(1, 3, 32, 32)
    >>> pipe = pipeline("ood-cifar10")
    >>> pipe(x)
    ```
    """

    return pipeline_registry[task](device=device, **kwargs)


from . import ood
