"""Detectors package."""

from detectors.utils import create_transform
from detectors.eval import get_ood_results
from . import config, data, eval, methods, models, utils
from .models import timm_register_model, create_model, list_models
from .data import create_dataset, register_dataset, get_dataset_cls, list_datasets
from .methods import create_detector, register_detector, list_detectors
from .pipelines import create_pipeline, register_pipeline, list_pipelines

__version__ = "0.1.0"
