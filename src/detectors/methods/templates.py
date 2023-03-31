"""Generalized OOD detection methods templates
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
import logging
from functools import partial
from typing import Callable, Dict, List, Literal, Optional
import detectors
import numpy as np
import torch
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from tqdm import tqdm
from .utils import create_reduction
from detectors.aggregations import create_aggregation

_logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Base class for OOD detectors."""

    def __init__(self, *args, **kwargs):
        """Initialize the detector."""
        pass

    def __call__(self, *args, **kwargs):
        """Run the detector."""
        raise NotImplementedError

    def __repr__(self):
        """Return the string representation of the detector."""
        return f"{self.__class__.__name__}()"

    def set_params(self, **params):
        """Set the parameters of the detector."""
        return self

    def save_params(self, path):
        """Save the parameters of the detector."""
        pass

    def load_params(self, path):
        """Load the parameters of the detector."""
        pass


class BaseDetectorWithFeatureExtraction:
    """Base class for OOD detectors with feature extraction."""

    def __init__(
        self,
        model: nn.Module,
        features_nodes: Optional[List[str]] = None,
        all_blocks: bool = False,
        last_layer: bool = False,
        pooling_op_name: Literal["max", "avg", "flatten", "getitem", "none"] = "avg",
        aggregation_method_name: Optional[str] = None,
        **kwargs,
    ):
        self.model = model
        self.features_nodes = features_nodes
        self.all_blocks = all_blocks
        self.pooling_op_name = pooling_op_name
        self.aggregation_method_name = aggregation_method_name

        # feature feature reduction operation
        self.reduction_op = create_reduction(self.pooling_op_name)

        if self.features_nodes is not None:
            # if features nodes were explicitly specified, use them
            pass
        elif hasattr(self.model, "feature_info") and all_blocks:
            # if all_blocks is True, use all blocks of the model
            self.features_nodes = [fi["module"] for fi in self.model.feature_info][1:]
        elif last_layer:
            # if last_layer is True, use the last layer of the model
            last_layer_name = list(self.model._modules.keys())[-1]
            if self.features_nodes is None:
                self.features_nodes = [last_layer_name]
            else:
                self.features_nodes.append(last_layer_name)
        else:
            # extract from the penultimate layer only
            self.features_nodes = [list(self.model._modules.keys())[-2]]
        _logger.info("Using features nodes: %s", self.features_nodes)

        self.features_extractor = create_feature_extractor(self.model, self.features_nodes)

        # insert reduction operation after each node
        def output_reduce(x: Dict[str, Tensor]):
            return {k: self.reduction_op(v) for k, v in x.items()}

        last_node = [n for n in self.features_extractor.graph.nodes if n.op == "output"][0]
        output_nodes = OrderedDict()
        output_nodes[last_node.name] = last_node.args[0]
        self.features_extractor.graph.erase_node(last_node)
        nodes = [n for n in self.features_extractor.graph.nodes]
        last_node = nodes[-1]
        with self.features_extractor.graph.inserting_after(last_node):
            new_node = self.features_extractor.graph.call_function(output_reduce, args=(output_nodes["output_1"],))
        nodes = [n for n in self.features_extractor.graph.nodes]
        with self.features_extractor.graph.inserting_after(nodes[-1]):
            self.features_extractor.graph.output(new_node)
        # self.features_extractor.graph.eliminate_dead_code()
        self.features_extractor.recompile()

        # multi-layer score aggregation
        self.aggregation_method = None
        if self.aggregation_method_name is not None:
            self.aggregation_method = create_aggregation(self.aggregation_method_name, **kwargs)

    @torch.no_grad()
    def start(self, example: Optional[Tensor] = None, fit_length: Optional[int] = None, *args, **kwargs):
        self.train_features = {}
        if example is not None and fit_length is not None:
            example_output = self.features_extractor(example)
            for node_name, v in example_output.items():
                self.train_features[node_name] = torch.zeros((fit_length,) + v.shape[1:], dtype=v.dtype)

    def update(self, x: Tensor):
        pass

    def end(self):
        pass

    def __call__(self, *args, **kwargs):
        """Run the detector."""
        raise NotImplementedError

    def __repr__(self):
        """Return the string representation of the detector."""
        return f"{self.__class__.__name__}()"

    def set_params(self, **params):
        """Set the parameters of the detector."""
        return self

    def save_params(self, path):
        """Save the parameters of the detector."""
        pass

    def load_params(self, path):
        """Load the parameters of the detector."""
        pass


def op_dict(x: Dict[str, Tensor]):
    op = partial(torch.mean, dim=(2, 3))
    return {k: op(v) for k, v in x.items()}


if __name__ == "__main__":
    model = detectors.create_model("resnet18", pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    detector = BaseDetectorWithFeatureExtraction(model, all_blocks=True)
    detector.start(x, 100)

    print(detector.reduction_op)
    print(detector.features_extractor.graph)
    print([v.shape for k, v in detector.features_extractor(x).items()])
