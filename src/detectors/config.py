"""
config.py
---------

This file contains the configuration for the project.
"""
import os

HOME = os.path.dirname(__file__)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HOME, "data/"))
IMAGENET_ROOT = os.environ.get("IMAGENET_ROOT", DATA_DIR)
CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", os.path.join(HOME, "checkpoints/"))
TENSORS_DIR = os.environ.get("TENSORS_DIR", os.path.join(HOME, "tensors/"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(HOME, "results/"))
