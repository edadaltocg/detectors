from abc import ABC, abstractmethod
import logging
from typing import Any


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
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def eval(self):
        pass