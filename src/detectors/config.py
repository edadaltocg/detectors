"""
config.py
---------

This file contains the configuration for the project.
"""
import os

DATA_DIR = os.environ.get("DATA_DIR", "data/")
IMAGENET_ROOT = os.environ.get("IMAGENET_ROOT", DATA_DIR)
CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")
TENSORS_DIR = os.environ.get("TENSORS_DIR", "tensors/")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results/")
