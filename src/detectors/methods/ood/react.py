import logging
from functools import partial
from typing import Callable, List, Optional

import torch
import torch.fx as fx
import torch.utils.data
import torchvision
import torchvision.models as models
from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

_logger = logging.getLogger(__name__)


def reactify(
    m: torch.nn.Module,
    condition_fn: Callable,
    insert_fn: Callable,
) -> torch.nn.Module:
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


@torch.no_grad()
def _compute_thresholds(feature_extractor, dataloader, device, limit=2000, p=0.9):
    scores = []
    feature_extractor.eval()
    counter = 0
    for x, _ in dataloader:
        x = x.to(device)
        outputs = feature_extractor(x)
        outputs = {k: v.detach().cpu() for k, v in outputs.items()}
        scores.append(outputs)
        counter += x.shape[0]
        if counter > limit:
            break

    scores = {k: torch.cat([s[k] for s in scores]) for k in scores[0].keys()}
    thrs = {k: torch.quantile(scores[k], p).item() for k in scores.keys()}
    return thrs


class ReAct:
    LIMIT = 2_560_000

    def __init__(
        self,
        model: torch.nn.Module,
        features_nodes: Optional[List[str]] = ["flatten"],
        graph_nodes_names: Optional[List[str]] = None,  # annoying parameter
        insert_node_fn: Callable = insert_fn,
        p=0.9,
        *args,
        **kwargs,
    ) -> None:
        self.model = model
        self.device = next(self.model.parameters()).device
        self.model.eval()
        self.features_nodes = features_nodes
        self.graph_nodes_names = graph_nodes_names
        if self.features_nodes is None:
            self.features_nodes = [self.model.feature_info[-1]["module"]]
        if self.graph_nodes_names is None and not hasattr(self.model, "forward_head"):
            raise ValueError(
                "You must pass graph_nodes_names if the model does not have forward_head attribute implemented."
            )
        if len(self.features_nodes) > 1 and self.graph_nodes_names is None:
            raise ValueError("The number of features nodes must be equal to the number of graph nodes.")
        self.feature_extractor = create_feature_extractor(self.model, self.features_nodes)
        self.insert_node_fn = insert_node_fn
        self.p = p

        self.thr = None
        self.training_features = {}

    def start(self, *args, **kwargs):
        self.training_features = {}

    def update(self, x: Tensor, y: Tensor) -> None:
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

        _logger.info(f"ReAct thresholds = {dict(zip(self.features_nodes, self.thrs))}")

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        self.model.eval()
        if self.graph_nodes_names is not None:
            logits = self.model(x)
        else:
            features = torch.clip(list(self.feature_extractor(x).values())[-1], max=self.thrs[-1])
            logits = self.model.forward_head(features)
        return torch.logsumexp(logits, dim=-1)
