"""
Generalized detection methods templates.
"""
import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Literal, Optional

import torch
import torch.distributed
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from detectors.aggregations import create_aggregation

from .utils import create_reduction

_logger = logging.getLogger(__name__)


def add_output_op(feature_extractor, output_op: Callable) -> nn.Module:
    last_node = [n for n in feature_extractor.graph.nodes if n.op == "output"][0]
    last_node_args = last_node.args
    feature_extractor.graph.erase_node(last_node)
    nodes = [n for n in feature_extractor.graph.nodes]
    with feature_extractor.graph.inserting_after(nodes[-1]):
        new_node = feature_extractor.graph.call_function(output_op, args=last_node_args)
    nodes = [n for n in feature_extractor.graph.nodes]
    with feature_extractor.graph.inserting_after(nodes[-1]):
        feature_extractor.graph.output(new_node)
    feature_extractor.recompile()
    return feature_extractor


class Detector(ABC):
    """Detector base class."""

    def __init__(self, **kwargs):
        pass

    def start(self, example: Optional[Tensor] = None, fit_length: Optional[int] = None, *args, **kwargs):
        """Setup detector for fitting parameters.

        Args:
            example (Optional[Tensor], optional): Input example. Useful for pre-allocating memory.
                Defaults to None.
            fit_length (Optional[int], optional): Length of the fitting dataset. Useful for pre-allocating memory.
                Defaults to None.

        This is called before the first call to `update` and is optional.
        """
        pass

    def update(self, x: Tensor, y: Tensor, *args, **kwargs):
        """Accumulate features for detector.

        Args:
            x (Tensor): input tensor.
            y (Tensor): target tensor.

        This is called for each batch of the fitting dataset and is optional.
        """
        pass

    def end(self, *args, **kwargs):
        """Finalize detector fitting process.

        This is called after the last call to `update` and is optional.
        """
        pass

    def fit(self, dataloader, **kwargs):
        """Fit detector to a dataset.

        Args:
            dataloader (Dataloader): Dataloader for the fitting dataset.
        """
        fit_length = len(dataloader.dataset)
        # get example
        x, y = next(iter(dataloader))
        self.start(example=x, fit_length=fit_length, **kwargs)
        for x, y in dataloader:
            self.update(x, y, **kwargs)
        self.end(**kwargs)
        return self

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """Compute scores for each input at test time.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: scores for each input.
        """
        raise NotImplementedError


class DetectorWrapper(Detector):
    """Detector interface."""

    def __init__(self, detector, **kwargs):
        self.detector = detector
        if hasattr(self.detector, "model"):
            self.model = self.detector.model
        elif hasattr(self.detector, "keywords") and "model" in self.detector.keywords:
            self.model = self.detector.keywords["model"]
        else:
            self.model = None
        self.keywords = kwargs

    def start(self, example: Optional[Tensor] = None, fit_length: Optional[int] = None, *args, **kwargs):
        if not hasattr(self.detector, "start"):
            _logger.warning("Detector does not have a start method.")
            return
        self.detector.start(example, fit_length, *args, **kwargs)

    def update(self, x: Tensor, y: Tensor, *args, **kwargs):
        if not hasattr(self.detector, "update"):
            _logger.warning("Detector does not have an update method.")
            return
        self.detector.update(x, y, *args, **kwargs)

    def end(self, *args, **kwargs):
        if not hasattr(self.detector, "end"):
            _logger.warning("Detector does not have an end method.")
            return
        self.detector.end(*args, **kwargs)

    def fit(self, dataloader, **kwargs):
        # get fit length # CHECK BUG
        fit_length = len(dataloader.dataset)
        # get example
        x, y = next(iter(dataloader))
        self.start(example=x, fit_length=fit_length, **kwargs)
        for x, y in dataloader:
            self.update(x, y, **kwargs)
        self.end(**kwargs)
        return self

    def __call__(self, x: Tensor) -> Tensor:
        return self.detector(x)

    def set_hyperparameters(self, **params):
        """Set the parameters of the detector."""
        model = params.pop("model", self.model)
        self.keywords.update(params)
        if hasattr(self.detector, "keywords"):
            self.detector.keywords.update(**params)
        else:
            self.detector = self.detector.__class__(model=model, **self.keywords)
        return self

    def save_params(self, path):
        """Save the parameters of the detector."""
        raise NotImplementedError

    def load_params(self, path):
        """Load the parameters of the detector."""
        raise NotImplementedError

    def __repr__(self):
        """Return the string representation of the detector."""
        return f"{self.__class__.__name__}()"


class DetectorWithFeatureExtraction(Detector):
    """Base class for OOD detectors with feature extraction.

    Args:
        model (nn.Module): Model to be used to extract features
        features_nodes (Optional[List[str]]): List of strings that represent the feature nodes.
            Defaults to None.
        all_blocks (bool, optional): If True, use all blocks of the model. Defaults to False.
        last_layer (bool, optional): If True, use also the last layer of the model. Defaults to False.
        pooling_op_name (str, optional): Pooling operation to be applied to the features.
            Can be one of:
                `max`, `avg`, `none`, `flatten`, `getitem`, `avg_or_getitem`, `max_or_getitem`.
            Defaults to "avg".
        aggregation_method_name (str, optional): Aggregation method to be applied to the features. Defaults to None.
        **kwargs
    """

    def __init__(
        self,
        model: nn.Module,
        features_nodes: Optional[List[str]] = None,
        all_blocks: bool = False,
        last_layer: bool = False,
        pooling_op_name: str = "avg",
        aggregation_method_name: Optional[str] = "mean",
        **kwargs,
    ):
        self.model = model
        self.features_nodes = features_nodes
        self.all_blocks = all_blocks
        self.pooling_op_name = pooling_op_name or "avg"
        self.aggregation_method_name = aggregation_method_name or "none"

        # feature feature reduction operation
        self.reduction_op = create_reduction(self.pooling_op_name)

        if self.features_nodes is not None:
            # if features nodes were explicitly specified, use them
            pass
        elif hasattr(self.model, "feature_info") and all_blocks:
            # if all_blocks is True, use all blocks of the model
            self.features_nodes = [fi["module"] for fi in self.model.feature_info][1:]  # type: ignore
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

        self.feature_extractor = create_feature_extractor(self.model, self.features_nodes)
        self.feature_extractor.eval()

        # insert reduction operation after each node
        def output_reduce(x: Dict[str, Tensor]):
            return {k: self.reduction_op(v) for k, v in x.items()}

        self.feature_extractor = add_output_op(self.feature_extractor, output_reduce)
        self.aggregation_method = create_aggregation(self.aggregation_method_name, **kwargs)

        self.train_features = {}
        self.train_targets = []
        self.idx = 0

    @torch.no_grad()
    def start(self, example: Optional[Tensor] = None, fit_length: Optional[int] = None, *args, **kwargs):
        self.train_features = {}
        self.train_targets = []
        self.idx = 0
        if example is not None and fit_length is not None:
            example_output = self.feature_extractor(example)
            for node_name, v in example_output.items():
                _logger.debug((fit_length,) + v.shape[1:])
                self.train_features[node_name] = torch.empty((fit_length,) + v.shape[1:], dtype=v.dtype)
                self.train_targets = torch.empty((fit_length,), dtype=torch.long)

    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor, *args, **kwargs):
        self.batch_size = x.shape[0]
        features: Dict[str, Tensor] = self.feature_extractor(x)
        # TODO: gather x and y?
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        for node_name, v in features.items():
            v = v.cpu()
            if node_name not in self.train_features:
                self.train_features[node_name] = [v]
            elif isinstance(self.train_features[node_name], list):
                self.train_features[node_name].append(v)
            else:
                self.train_features[node_name][self.idx : self.idx + v.shape[0]] = v

        y = y.cpu()
        if isinstance(self.train_targets, list):
            self.train_targets.append(y)
        else:
            self.train_targets[self.idx : self.idx + y.shape[0]] = y

        self.idx += x.shape[0]

    def end(self, *args, **kwargs):
        for node_name, v in self.train_features.items():
            if isinstance(v, list):
                self.train_features[node_name] = torch.cat(v, dim=0)
        if isinstance(self.train_targets, list):
            self.train_targets = torch.cat(self.train_targets, dim=0)

        self._fit_params()

        if self.aggregation_method is not None and hasattr(self.aggregation_method, "fit"):
            _logger.info("Fitting aggregator %s...", self.aggregation_method_name)
            self.batch_size = self.train_targets.shape[0]  # type: ignore
            all_scores = torch.zeros(self.train_targets.shape[0], len(self.train_features))
            for i, (k, v) in tqdm(enumerate(self.train_features.items())):
                idx = 0
                for idx in range(0, v.shape[0], self.batch_size):
                    all_scores[:, i] = self._layer_score(v[idx : idx + self.batch_size], k, i).view(-1)
            self.aggregation_method.fit(all_scores, self.train_targets)

        # TODO: compile graph with _layer_score

        del self.train_features
        del self.train_targets

    @abstractmethod
    def _fit_params(self) -> None:
        """Fit the data to the parameters of the detector."""
        pass

    @abstractmethod
    def _layer_score(self, x: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None):
        """Compute the anomaly score for a single layer.

        Args:
            x (Tensor): input tensor.
            layer_name (str, optional): name of the layer. Defaults to None.
            index (int, optional): index of the layer in the feature extractor. Defaults to None.
        """
        raise NotImplementedError

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        features = self.feature_extractor(x)
        all_scores = torch.zeros(x.shape[0], len(features), device=x.device)
        for i, (k, v) in enumerate(features.items()):
            all_scores[:, i] = self._layer_score(v, k, i).view(-1)

        all_scores = self.aggregation_method(all_scores)
        return all_scores.view(-1)
