import logging
from functools import partial
from typing import Callable

import torch
import torch.fx as fx
import torch.utils.data
import torchvision
import torchvision.models as models
from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


logger = logging.getLogger(__name__)


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
def compute_thresholds(feature_extractor, dataloader, device, limit=2000, p=0.9):
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
    def __init__(
        self,
        model: torch.nn.Module,
        penultimate_node: str = "flatten",
        insert_node_fn: Callable = insert_fn,
        p=0.9,
        *args,
        **kwargs
    ) -> None:
        self.model = model
        self.penultimate_node = penultimate_node
        self.feature_extractor = create_feature_extractor(model, [penultimate_node])
        self.insert_node_fn = insert_node_fn
        self.p = p

        self.thr = None
        self.all_training_features = {}

    def fit(self, x: Tensor, y: Tensor) -> None:
        with torch.no_grad():
            features = self.feature_extractor(x)

        # accumulate training features
        if len(self.all_training_features) == 0:
            for k in features:
                self.all_training_features[k] = features[k].cpu()
        else:
            for k in features:
                self.all_training_features[k] = torch.cat((self.all_training_features[k], features[k].cpu()), dim=0)

    def on_fit_end(self, *args, **kwargs):
        thrs = {
            k: torch.quantile(self.all_training_features[k], self.p).item() for k in self.all_training_features.keys()
        }
        self.thr = thrs[self.penultimate_node]
        self.model = reactify(
            self.model,
            condition_fn=partial(condition_fn, equals_to=self.penultimate_node),
            insert_fn=partial(insert_fn, thr=self.thr),
        )
        logger.info(f"ReAct: threshold = {self.thr:.4f}")

    def __call__(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            logits = self.model(x)
        return torch.logsumexp(logits, dim=-1)


def test():
    model = models.resnet18()
    print(get_graph_node_names(model)[0])
    transformed_model = reactify(model, partial(condition_fn, equals_to="flatten"), partial(insert_fn, thr=1.0))
    print(transformed_model.graph)
    print(transformed_model.code)

    x = torch.randn(5, 3, 224, 224)
    torch.allclose(model(x), transformed_model(x))

    feature_extractor = create_feature_extractor(model, ["flatten"])
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    thrs = compute_thresholds(feature_extractor, dataloader, device="cpu", limit=100, p=0.7)
    print(thrs)
    react_model = reactify(model, partial(condition_fn, equals_to="flatten"), partial(insert_fn, thr=thrs["flatten"]))
    print(react_model)
    print(react_model(x).max(), model(x).max())
    assert not torch.allclose(react_model(x), model(x))


if __name__ == "__main__":
    test()
