from functools import partial
from typing import Callable, Dict, List

import torch
import torch.fx as fx
from torchvision.models.feature_extraction import get_graph_node_names

from detectors.methods.projection import Projection
from detectors.methods.react import condition_fn, insert_fn, reactify


class ReActProjection(Projection):
    def __init__(
        self,
        model: torch.nn.Module,
        features_nodes: List[str] = ["layer1", "layer2", "layer3", "layer4", "clip", "fc"],
        pooling_name: str = "max",
        graph_nodes_names_thr: Dict[str, float] = {"global_pool_flatten": 1.0},
        insert_node_fn: Callable = insert_fn,
        **kwargs,
    ):
        self.graph_nodes_names_thr = graph_nodes_names_thr
        self.insert_node_fn = insert_node_fn
        for node_name, thr in self.graph_nodes_names_thr.items():
            model = reactify(
                model,
                condition_fn=partial(condition_fn, equals_to=node_name),
                insert_fn=partial(self.insert_node_fn, thr=thr),
            )

        super().__init__(model, features_nodes, pooling_name, **kwargs)


def test():
    import torchvision.models as models

    model = models.densenet121()
    print(get_graph_node_names(model)[0])
    graph: fx.Graph = fx.Tracer().trace(model)
    model = fx.GraphModule(model, graph)
    print(model.code)
    graph_nodes_names_thr = {
        "features_transition1_pool": 1.0,
        "features_transition2_pool": 1.0,
        "features_transition3_pool": 1,
        "features_norm5": 1,
        "flatten": 1,
    }
    for node_name, thr in graph_nodes_names_thr.items():
        model = reactify(model, partial(condition_fn, equals_to=node_name), partial(insert_fn, thr=thr))
    print(model.graph)
    print(model.code)

    model.classifier = torch.nn.Linear(1024, 3)
    model.eval()
    x = torch.rand(32, 3, 224, 224)
    y = torch.randint(0, 3, (32,))
    projection = ReActProjection(
        model, ["clip", "clip_1", "clip_2", "clip_3", "clip_4", "classifier"], "max", graph_nodes_names_thr
    )
    projection.fit(x, y)
    projection.end()
    print(projection(x))
    assert projection(x).shape == (32,)


if __name__ == "__main__":
    test()
