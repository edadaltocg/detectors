"""
Configuration for the project.

It is used to set the default paths for the data, checkpoints, results, etc.

Constants:

- `DATA_DIR`: The directory where the data is stored.
- `IMAGENET_ROOT`: The directory where the ImageNet data is stored.
- `CHECKPOINTS_DIR`: The directory where the checkpoints are stored.
- `RESULTS_DIR`: The directory where the results are stored.
"""
import os

HOME = os.path.dirname(__file__)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HOME, "data/"))
IMAGENET_ROOT = os.environ.get("IMAGENET_ROOT", DATA_DIR)
CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", os.path.join(HOME, "checkpoints/"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(HOME, "results/"))
