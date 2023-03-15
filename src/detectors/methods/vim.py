from sklearn.covariance import EmpiricalCovariance

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor
import logging
import torch.distributed as dist

_logger = logging.getLogger(__name__)


class ViM:
    """Virtual Logit Matching (ViM)

    - Title: ViM: Out-Of-Distribution with Virtual-logit Matching.
    - Paper: [https://arxiv.org/abs/2203.10807](https://arxiv.org/abs/2203.10807)
    - GitHub: [https://github.com/haoqiwang/vim/](https://github.com/haoqiwang/vim/)
    """

    def __init__(self, model: nn.Module, last_layer_name: str = None, penultimate_layer_name: str = None, **kwargs):
        self.model = model
        self.last_layer_name = last_layer_name
        self.penultimate_layer_name = penultimate_layer_name

        # create feature extractor
        if self.penultimate_layer_name is None:
            self.penultimate_layer_name = list(self.model._modules.keys())[-2]
        self.feature_extractor = create_feature_extractor(self.model, [self.penultimate_layer_name])

        # get the model weights of the last layer
        if last_layer_name is None:
            last_layer_name = list(model._modules.keys())[-1]
        last_layer = model._modules[last_layer_name]
        assert isinstance(last_layer, nn.Linear), "Last layer must be a linear layer"

        self.w = last_layer.weight.data.clone()
        self.b = last_layer.bias.data.clone()

        # new origin
        self.u = -torch.matmul(torch.linalg.pinv(self.w), self.b).float()
        _logger.debug("New origin: %s", self.u)

        self.principal_subspace = None
        self.train_features = []
        self.train_logits = []
        self.alpha = None
        self.top_k = None

    def _get_logits(self, features: Tensor) -> Tensor:
        logits = torch.matmul(features, self.w.T.to(features.device)) + self.b.to(features.device)
        return logits

    def start(self):
        self.principal_subspace = None
        self.train_features = []
        self.train_logits = []
        self.alpha = None
        self.top_k = None

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        features = self.feature_extractor(x)[self.penultimate_layer_name]

        if features.ndim == 4:
            # avg pooling if necessary
            features = features.mean(dim=[2, 3])

        if dist.is_initialized():
            dist.gather(features, dst=0)

        self.train_features.append(features.detach().cpu())
        self.train_logits.append(self._get_logits(features).detach().cpu())

    def end(self):
        self.train_features = torch.cat(self.train_features)
        self.train_logits = torch.cat(self.train_logits)
        self.top_k = 1000 if self.train_features.shape[1] > 1500 else 512

        _logger.info("Train features shape: %s", self.train_features.shape)

        # calculate eigenvectors of the covariance matrix
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(self.train_features.detach().cpu().numpy() - self.u.detach().cpu().numpy())
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        determinant = np.linalg.det(ec.covariance_)
        _logger.debug("Determinant: %s", determinant)
        _logger.debug("Eigen values: %s", eig_vals)
        # cov_mat=torch.cov(self.train_features.cpu() - self.u.cpu())
        # eig_vals, eigen_vectors = torch.linalg.eig(cov_mat)

        # select largest eigenvectors to get the principal subspace
        largest_eigvals_idx = np.argsort(eig_vals * -1)[self.top_k :]
        self.principal_subspace = torch.from_numpy(
            np.ascontiguousarray((eigen_vectors.T[largest_eigvals_idx]).T)
        ).float()
        _logger.debug("Principal subspace: %s", self.principal_subspace)

        # calculate residual
        x_p_t = torch.matmul(self.train_features.cpu() - self.u.cpu(), self.principal_subspace.cpu())
        vlogits = torch.norm(x_p_t, dim=-1)
        self.alpha = self.train_logits.max(dim=-1)[0].mean() / vlogits.mean()
        _logger.debug("Alpha: %s", self.alpha)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.feature_extractor(x)[self.penultimate_layer_name]

        logits = self._get_logits(features)

        x_p_t = torch.norm(torch.matmul(features - self.u.to(x.device), self.principal_subspace.to(x.device)), dim=-1)
        vlogit = x_p_t * self.alpha
        energy = torch.logsumexp(logits, dim=-1)
        score = -vlogit + energy
        _logger.debug("Score: %s", score)
        return score
