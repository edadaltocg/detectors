from typing import Callable, Optional

from detectors.pipelines import register_pipeline
from detectors.pipelines.ood import OODBenchmarkPipeline


@register_pipeline("osr_cifar10")
class OSRCifar10(OODBenchmarkPipeline):
    def __init__(
        self, transform: Callable, batch_size: int, limit_fit: Optional[int] = None, seed: int = 42, **kwargs
    ) -> None:
        super().__init__("cifar10", {"cifar100": "test"}, transform, batch_size, limit_fit, seed)


@register_pipeline("osr_cifar100")
class OSRCifar100(OODBenchmarkPipeline):
    def __init__(
        self, transform: Callable, batch_size: int, limit_fit: Optional[int] = None, seed: int = 42, **kwargs
    ) -> None:
        super().__init__("cifar100", {"cifar10": "test"}, transform, batch_size, limit_fit, seed)
