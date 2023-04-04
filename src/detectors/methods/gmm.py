import logging
from typing import List, Literal, Optional

import torch
from torch import Tensor, nn

from detectors.methods.gmm_torch import GaussianMixture
from detectors.methods.templates import DetectorWithFeatureExtraction

_logger = logging.getLogger(__name__)


class GMM(DetectorWithFeatureExtraction):
    def __init__(
        self,
        model: nn.Module,
        features_nodes: Optional[List[str]] = None,
        all_blocks: bool = False,
        last_layer: bool = False,
        pooling_op_name: str = "avg",
        aggregation_method_name: str = "mean",
        n_components: Optional[int] = None,
        covariance_type: Literal["full", "tied", "diag"] = "full",
        **kwargs_gmm
    ):
        super().__init__(
            model,
            features_nodes=features_nodes,
            all_blocks=all_blocks,
            last_layer=last_layer,
            pooling_op_name=pooling_op_name,
            aggregation_method_name=aggregation_method_name,
        )
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.kwargs_gmm = kwargs_gmm

    def _layer_score(self, x: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None):
        return self.gms[layer_name].score_samples(x).view(-1)

    def _fit_params(self) -> None:
        _logger.info("Estimating GMM parameters...")

        # estimate GMM parameters
        if self.n_components is None:
            self.n_components = torch.unique(self.train_targets).shape[0]
        _logger.info("Number of components set to %i.", self.n_components)

        self.gms = {}
        device = next(self.model.parameters()).device
        for layer_name, layer_features in self.train_features.items():
            self.gms[layer_name] = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                init_params="random_from_data",
                **self.kwargs_gmm
            )
            self.gms[layer_name].fit(layer_features.to(device))
