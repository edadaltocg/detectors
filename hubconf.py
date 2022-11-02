import os
import sys


dependencies = ["torch"]


SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(SRC_DIR)

from detectors.models.resnet import resnet18_cifar10, resnet34_cifar10
