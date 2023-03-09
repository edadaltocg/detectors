import logging
from functools import partial
from typing import List, Literal, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.models.feature_extraction import create_feature_extractor


class OpenMax:
    """
    Implementation of the OpenMax Layer as proposed in the paper *Towards Open Set Deep Networks*.

    The methods determines a center :math:`\\mu_y` for each class in the logits space of a model, and then
    creates a statistical model of the distances of correct classified inputs.
    It uses extreme value theory to detect outliers by fitting a weibull function to the tail of the distance
    distribution.

    We use the activation of the *unknown* class as outlier score.

    .. warning:: This methods requires ``libmr`` to be installed, which is broken at the moment. You can only use it
       by installing ``cython`` and ``numpy``, and ``libmr`` manually afterwards.

    :see Paper: `ArXiv <https://arxiv.org/abs/1511.06233>`__
    :see Implementation: `GitHub <https://github.com/abhijitbendale/OSDN>`__
    """

    def __init__(
        self,
        model: nn.Module,
        tailsize: int = 25,
        alpha: int = 10,
        euclid_weight: float = 1.0,
    ):
        """
        :param model: neural network, assumed to output logits
        :param tailsize: length of the tail to fit the distribution to
        :param alpha: number of class activations to revise
        :param euclid_weight: weight for the euclidean distance.
        """
        self.model = model

    def fit(self, data_loader: DataLoader, device: Optional[str] = "cpu"):
        """
        Determines parameters of the weibull functions for each class.

        :param data_loader: Data to use for fitting
        :param device: Device used for calculations
        """
        z, y = OpenMax._extract(data_loader, self.model, device=device)
        self.openmax.fit(z.numpy(), y.numpy())
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input, will be passed through the model to obtain logits
        """
        with torch.no_grad():
            z = self.model(x).cpu().numpy()

        return torch.tensor(self.openmax.predict(z)[:, 0])

    @staticmethod
    def _extract(data_loader, model: torch.nn.Module, device):
        """
        Extract embeddings from model. Ignores OOD data.
        """
        buffer = TensorBuffer()
        log.debug("Extracting features")
        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            known = is_known(y)
            z = model(x[known])
            # flatten
            x = z.view(x.shape[0], -1)
            buffer.append("embedding", z)
            buffer.append("label", y[known])

        z = buffer.get("embedding")
        y = buffer.get("label")

        buffer.clear()
        return z, y
