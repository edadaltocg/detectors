import logging
from typing import List, Literal, Optional

import torch
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor
import torch.distributed as dist

from detectors.methods.gmm_torch import GaussianMixture

from .utils import create_reduction

_logger = logging.getLogger(__name__)


class GMM:
    def __init__(
        self,
        model: nn.Module,
        features_nodes: Optional[List[str]] = None,
        pooling_op_name: str = "avg",
        n_components: Optional[int] = None,
        **kwargs_gmm
    ):
        self.model = model
        self.model.eval()
        self.features_nodes = features_nodes
        if self.features_nodes is None:
            self.features_nodes = [list(self.model._modules.keys())[-2]]
        self.feature_extractor = create_feature_extractor(self.model, self.features_nodes)

        self.pooling_op = create_reduction(pooling_op_name)
        self.n_components = n_components
        self.kwargs_gmm = kwargs_gmm

        self.gms = {}
        self.training_features = {}
        self.targets = None

    def start(self):
        self.gms = {}
        self.training_features = {}
        self.targets = None

    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor) -> None:
        self.feature_extractor = self.feature_extractor.to(x.device)
        features = self.feature_extractor(x)
        if isinstance(features, Tensor):
            features = {"penultimate": self.feature_extractor(x)}

        for k in features:
            features[k] = self.pooling_op(features[k])
            # dist torch accumulate
            if dist.is_initialized():
                dist.gather(features, dst=0)

        # accumulate training features
        for k in features:
            if len(self.training_features) == 0:
                self.training_features[k] = features[k]
                self.targets = y
            else:
                self.training_features[k] = torch.cat((self.training_features[k], features[k]), dim=0)
                self.targets = torch.cat((self.targets, y), dim=0)

    def end(self):
        _logger.info("Estimating GMM parameters...")

        # estimate GMM parameters
        if self.n_components is None:
            # self.n_components = np.unique(self.targets).shape[0]
            self.n_components = torch.unique(self.targets).shape[0]
            _logger.info("Number of components set to %i.", self.n_components)

        for k in self.training_features:
            _logger.info("Training features shape: %s", self.training_features[k].shape)
            self.gms[k] = GaussianMixture(
                n_components=self.n_components,
                covariance_type="full",
                init_params="random_from_data",
                tol=1e-3,
                reg_covar=1e-4,
                **self.kwargs_gmm
            )
            self.gms[k].fit(self.training_features[k])

    def __call__(self, x: Tensor) -> Tensor:
        self.feature_extractor = self.feature_extractor.to(x.device)
        with torch.no_grad():
            features = self.feature_extractor(x)
            if isinstance(features, Tensor):
                features = {"penultimate": self.feature_extractor(x)}

        for k in features:
            features[k] = self.pooling_op(features[k])

        stack = None
        for k in features:
            scores = self.gms[k].score_samples(features[k])
            scores = scores.view(-1, 1)
            if stack is None:
                stack = scores
            else:
                stack = torch.cat((stack, scores), dim=1)  # type: ignore

        if stack is None:
            raise ValueError("Stack is None, this should not happen.")

        return stack.mean(dim=1)
