from typing import Callable, List, Optional

import torch
from torch import Tensor, nn

from detectors.methods.templates import DetectorWithFeatureExtraction


def fr_distance_univariate_gaussian(
    mu_1: torch.Tensor, sig_1: torch.Tensor, mu_2: torch.Tensor, sig_2: torch.Tensor
) -> torch.Tensor:
    """Calculates the Fisher-Rao distance between univariate gaussian distributions in prallel.

    Args:
        mu_1 (torch.Tensor): Tensor of dimension (N,*) containing the means of N different univariate gaussians
        sig_1 (torch.Tensor): Standard deviations of univariate gaussian distributions
        mu_2 (torch.Tensor): Means of the second univariate gaussian distributions
        sig_2 (torch.Tensor): Standard deviation of the second univariate gaussian distributions
    Returns:
        torch.Tensor: Distance tensor of size (N,*)
    """
    dim = len(mu_1.shape)
    mu_1, mu_2 = mu_1.reshape(*mu_1.shape, 1), mu_2.reshape(*mu_2.shape, 1)
    sig_1, sig_2 = sig_1.reshape(*sig_1.shape, 1), sig_2.reshape(*sig_2.shape, 1)

    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=mu_1.device))
    a = torch.norm(
        torch.cat((mu_1 / sqrt_2, sig_1), dim=dim) - torch.cat((mu_2 / sqrt_2, -1 * sig_2), dim=dim), p=2, dim=dim
    )
    b = torch.norm(
        torch.cat((mu_1 / sqrt_2, sig_1), dim=dim) - torch.cat((mu_2 / sqrt_2, sig_2), dim=dim), p=2, dim=dim
    )

    num = a + b + 1e-12
    den = a - b + 1e-12
    return sqrt_2 * torch.log(num / den)


def fr_distance_multivariate_gaussian(
    x: torch.Tensor, y: torch.Tensor, cov_x: torch.Tensor, cov_y: torch.Tensor
) -> torch.Tensor:
    num_examples_x = x.shape[0]
    num_examples_y = y.shape[0]
    # Replicate std dev. matrix to match the batch size
    sig_x = torch.vstack([torch.sqrt(torch.diag(cov_x)).reshape(1, -1)] * num_examples_x)
    sig_y = torch.vstack([torch.sqrt(torch.diag(cov_y)).reshape(1, -1)] * num_examples_y)
    return torch.sqrt(torch.sum(fr_distance_univariate_gaussian(x, sig_x, y, sig_y) ** 2, dim=1)).reshape(-1, 1)


def _igeood_layer_score(x, mus, cov_x, cov_mus):
    if type(mus) == dict:
        mus = torch.vstack([mu.reshape(1, -1) for mu in mus.values()])
    else:
        mus = [mus.reshape(1, -1)]

    stack = torch.hstack(
        [fr_distance_multivariate_gaussian(x, mu.reshape(1, -1), cov_x, cov_mus).reshape(-1, 1) for mu in mus]
    )
    return stack


def igeood_layer_score_min(x, mus, cov_x, cov_mus):
    stack = _igeood_layer_score(x, mus, cov_x, cov_mus)
    return torch.min(stack, dim=1)[0]


def class_cond_mus_cov_matrix(x: Tensor, targets: Tensor, device=torch.device("cpu")):
    class_cond_mean = {}
    centered_data_per_class = {}
    unique_classes = sorted(torch.unique(targets.detach().cpu()).numpy().tolist())
    for c in unique_classes:
        filt = targets == c
        temp = x[filt].to(device)
        class_cond_mean[c] = temp.mean(0, keepdim=True)
        centered_data_per_class[c] = temp - class_cond_mean[c]

    centered_data_per_class = torch.vstack(list(centered_data_per_class.values()))
    mus = torch.vstack(list(class_cond_mean.values()))

    cov_mat = torch.matmul(centered_data_per_class.T, centered_data_per_class) / centered_data_per_class.shape[0]
    cov_mat = torch.diag(torch.diag(cov_mat))
    return mus, cov_mat


HYPERPARAMETERS = dict()


class IgeoodFeatures(DetectorWithFeatureExtraction):
    """Igeood OOD detector.

    Args:
        model (nn.Module): Model to be used to extract features
        features_nodes (Optional[List[str]]): List of strings that represent the feature nodes.
            Defaults to None.
        all_blocks (bool, optional): If True, use all blocks of the model. Defaults to False.
        last_layer (bool, optional): If True, use also the last layer of the model. Defaults to False.
        pooling_op_name (str, optional): Pooling operation to be applied to the features.
            Can be one of `max`, `avg`, `flatten`, `getitem`, `avg_or_getitem`, `max_or_getitem`, `none`. Defaults to `avg`.
        aggregation_method_name (str, optional): Aggregation method to be applied to the features. Defaults to None.
        mu_cov_est_fn (Callable, optional): Function to estimate the mean and covariance matrix of the features.

    References:
        [1] https://arxiv.org/abs/2203.07798
    """

    def __init__(
        self,
        model: nn.Module,
        features_nodes: Optional[List[str]] = None,
        all_blocks: bool = False,
        last_layer: bool = False,
        pooling_op_name: str = "avg_or_getitem",
        aggregation_method_name: Optional[str] = "mean",
        mu_cov_est_fn: Callable = class_cond_mus_cov_matrix,
        **kwargs,
    ):
        super().__init__(
            model, features_nodes, all_blocks, last_layer, pooling_op_name, aggregation_method_name, **kwargs
        )
        self.mu_cov_est_fn = mu_cov_est_fn

    def _layer_score(self, x: Tensor, layer_name: Optional[str] = None, index: Optional[int] = None):
        return igeood_layer_score_min(
            x,
            self.mus[layer_name].to(x.device),
            self.cov_mats[layer_name].to(x.device),
            self.cov_mats[layer_name].to(x.device),
        )

    def _fit_params(self) -> None:
        self.mus = {}
        self.cov_mats = {}
        device = next(self.model.parameters()).device
        for layer_name, layer_features in self.train_features.items():
            self.mus[layer_name], self.cov_mats[layer_name] = self.mu_cov_est_fn(
                layer_features, self.train_targets, device=device
            )
