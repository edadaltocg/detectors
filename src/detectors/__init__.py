"""Detectors package."""

from detectors.eval import get_ood_results

from . import config, data, eval, methods, models, utils
from .data import create_dataset, get_dataset_cls, list_datasets, register_dataset
from .methods import create_detector, create_hyperparameters, list_detectors, register_detector
from .methods.templates import Detector
from .models import create_model, create_transform, list_models, timm_register_model
from .pipelines import create_pipeline, list_pipelines, register_pipeline
