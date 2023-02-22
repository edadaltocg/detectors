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

    def postprocess(self, *args, **kwargs):
        return

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def report(cls, *args, **kwargs) -> str:
        raise NotImplementedError
