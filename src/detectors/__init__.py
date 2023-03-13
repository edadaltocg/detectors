"""Detectors package."""
from timm import create_model

from detectors.utils import create_transform

from . import config, data, eval, methods, models, utils
from .data import create_dataset, get_dataset_cls, list_datasets
from .methods import create_detector
from .pipelines import create_pipeline

__version__ = "0.1.0"
