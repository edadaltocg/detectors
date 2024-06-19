"""Base abstract pipeline class."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Pipeline:
    """Base Pipeline class."""

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

    def report(self, *args, **kwargs) -> str:
        raise NotImplementedError
