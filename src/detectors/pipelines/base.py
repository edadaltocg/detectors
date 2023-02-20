import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class Pipeline(ABC):
    def __init__(self) -> None:
        self.setup()

    def save_pretrained(self, path):
        raise NotImplementedError

    def load_pretrained(self, path):
        raise NotImplementedError

    def setup(self, *args, **kwargs):
        return

    def preprocess(self, *args, **kwargs) -> Any:
        return

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def postprocess(self, *args, **kwargs):
        return

    def benchmark(self, method: Callable) -> Dict[str, Any]:
        raise NotImplementedError

    def report(self):
        raise NotImplementedError
