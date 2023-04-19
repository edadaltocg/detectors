import logging
from functools import partial
from typing import Callable, List, Optional

import torch
import torch.fx as fx
import torch.utils.data
from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor

_logger = logging.getLogger(__name__)

HYPERPARAMETERS = dict(p=dict(low=0.1, high=1.0, type=float, default=0.9, step=0.05))


def reactify(m: torch.nn.Module, condition_fn: Callable, insert_fn: Callable) -> torch.nn.Module:
    graph: fx.Graph = fx.Tracer().trace(m)
    # Transformation logic here
    for node in graph.nodes:
        if condition_fn(node):
            insert_fn(node, graph)

    # Return new Module
    return fx.GraphModule(m, graph)


def condition_fn(node, equals_to: str):
    if node.name == equals_to:
        return True
    return False


def insert_fn(node, graph: fx.Graph, thr: float = 1.0):
    with graph.inserting_after(node):
        new_node = graph.call_function(torch.clip, args=(node,), kwargs={"max": thr})

        # change inputs of the next node and keep the input from the new node
        node.replace_all_uses_with(new_node)
        new_node.replace_input_with(new_node, node)


class ReAct:
    """ReAct detector.

    Args:
        model (torch.nn.Module): Model to be used to extract features
        features_nodes (Optional[List[str]]): List of strings that represent the feature nodes.
            Defaults to None.
        graph_nodes_names (Optional[List[str]]): List of strings that represent the graph nodes.
            Defaults to None.
        insert_node_fn (Callable): Function to be used to insert the node. Defaults to insert_fn.
        p (float, optional): Threshold to be used to clip the features. Defaults to 0.9.

    References:
        [1] https://arxiv.org/abs/2111.12797
    """

    LIMIT = 2_560_000

    def __init__(
        self,
        model: torch.nn.Module,
        features_nodes: Optional[List[str]] = None,
        graph_nodes_names: Optional[List[str]] = None,
        insert_node_fn: Callable = insert_fn,
        p=0.9,
        **kwargs,
    ) -> None:
        self.model = model
        self.device = next(self.model.parameters()).device
        self.model.eval()
        self.features_nodes = features_nodes
        self.graph_nodes_names = graph_nodes_names
        if self.features_nodes is None:
            self.features_nodes = [list(self.model._modules.keys())[-2]]
        self.feature_extractor = create_feature_extractor(self.model, self.features_nodes)

        self.last_layer = list(self.model._modules.values())[-1]

        self.insert_node_fn = insert_node_fn
        self.p = p

        self.thr = None
        self.training_features = {}

    def start(self, *args, **kwargs):
        self.training_features = {}

    def update(self, x: Tensor, y: Tensor) -> None:
        self.device = x.device
        self.feature_extractor = self.feature_extractor.to(x.device)
        if len(self.training_features.keys()) > 0:
            k = list(self.training_features.keys())[0]
            if self.training_features[k].view(-1).shape[0] > self.LIMIT:
                return

        with torch.no_grad():
            features = self.feature_extractor(x)

        # accumulate training features
        if len(self.training_features) == 0:
            for k in features:
                self.training_features[k] = features[k].cpu()
        else:
            for k in features:
                self.training_features[k] = torch.cat((self.training_features[k], features[k].cpu()), dim=0)

    def end(self, *args, **kwargs):
        self.thrs = list(
            {
                k: torch.quantile(self.training_features[k].view(-1)[: self.LIMIT].to(self.device), self.p).item()
                for k in self.training_features.keys()
            }.values()
        )
        if self.graph_nodes_names is not None:
            for i, node_name in enumerate(self.graph_nodes_names):
                # add clipping node to every feature node in the graph passed in the constructor
                self.model = reactify(
                    self.model,
                    condition_fn=partial(condition_fn, equals_to=node_name),
                    insert_fn=partial(insert_fn, thr=self.thrs[i]),
                )

        _logger.info("ReAct thresholds = %s", dict(zip(self.features_nodes, self.thrs)))

        del self.training_features

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        if self.graph_nodes_names is not None:
            self.model = self.model.to(x.device)
            logits = self.model(x)
        else:
            self.feature_extractor = self.feature_extractor.to(x.device)
            features = torch.clip(list(self.feature_extractor(x).values())[-1], max=self.thrs[-1])
            logits = self.last_layer(features)  # type: ignore
        return torch.logsumexp(logits, dim=-1)
