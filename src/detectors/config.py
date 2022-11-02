import os


DATASETS_DIR = os.environ.get("DATASETS_DIR", "data/")
IMAGENET_ROOT = os.environ.get("IMAGENET_ROOT", DATASETS_DIR)
CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")
TENSORS_DIR = os.environ.get("TENSORS_DIR", "tensors/")

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results/")

os.environ["TORCH_HOME"] = CHECKPOINTS_DIR
