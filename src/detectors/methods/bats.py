"""Typical Feature Estimated Method (TFEM)"""
import logging
from functools import partial
from typing import List, Literal, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from detectors.aggregations import create_aggregation

from .utils import create_reduction

_logger = logging.getLogger(__name__)


import torch


class BATS:
    def __init__(self, lbd=1) -> None:
        self.lbd = lbd

    def start(self):
        return

    def update(self, x: Tensor, y: Tensor):
        return

    def end(self):
        return

    def __call__(self, x: Tensor) -> Tensor:
        return torch.ones(x.shape[0])
