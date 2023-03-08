"""Detectors package."""
from . import config, data, eval, models, utils
from .data import create_dataset, get_dataset_cls, list_datasets
from .methods.ood import create_ood_detector
from .pipelines import create_pipeline

__version__ = "0.1.0"
