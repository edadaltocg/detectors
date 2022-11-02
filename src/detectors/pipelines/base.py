import logging
from abc import ABC, abstractmethod
from typing import Any, Tuple


logger = logging.getLogger(__name__)


class Pipeline(ABC):
    def __init__(self) -> None:
        pass

    def save_pretrained(self, path):
        raise NotImplementedError

    def preprocess(self, text):
        raise NotImplementedError

    def forward(self, text):
        raise NotImplementedError

    def postprocess(self, text):
        raise NotImplementedError

    def _setup(self):
        raise NotImplementedError

    def benchmark(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any):
        raise NotImplementedError
