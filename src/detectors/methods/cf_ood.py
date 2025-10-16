import logging
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

import detectors.methods.templates as detector_classes

logger = logging.getLogger(__name__)


class CFOOD_Detector(detector_classes.DetectorWithFeatureExtraction):

    def __init__(
        self,
        model: nn.Module,
        features_nodes: Optional[List[str]] = None,
        all_blocks: bool = False,
        last_layer: bool = False,
        pooling_op_name: str = "none",
        aggregation_method_name: str = "none",
        alpha: float = 0.1,  # your requested defaults
        k: float = 0.01,  # your requested defaults
        avg_topk: bool = False,
        **kwargs,
    ):
        super().__init__(
            model,
            features_nodes=features_nodes,
            all_blocks=all_blocks,
            last_layer=last_layer,
            pooling_op_name=pooling_op_name,
            aggregation_method_name=aggregation_method_name,
        )

        assert 0 < alpha <= 1.0, "alpha must be in (0, 1]"
        assert 0.0 <= k <= 1.0, "k must be in [0, 1]"

        self.alpha = alpha
        self.k = k
        self.mean_op = avg_topk

        # Per-layer state
        self.ref = {}  # normalized reference features: Tensor [n, d]
        self.index = {}  # selected indices: Tensor [n]

        # Candidate k-values; their positions depend on n and are set in _fit_params
        self.kval = np.concatenate(
            (np.linspace(0.001, 0.1, 100), np.linspace(0.105, 0.5, 81), np.linspace(0.51, 0.99, 49))
        )
        self.kpos = None
        self.klen = None
        self.knndst = None  # thresholds of kNN distances: Tensor [n, klen] (single-layer storage)

    @torch.no_grad()
    def _fit_params(self) -> None:
        """
        Build the reference set and the matrix of kNN distance thresholds for each layer.
        """
        self.ref = {}
        self.index = {}

        for layer_name, layer_features in self.train_features.items():
            # Flatten to [N, D]
            f = layer_features
            if f.dim() > 2:
                f = f.reshape(f.shape[0], -1)
            n_samples = f.shape[0]
            logger.info("Number of training samples for %s: %d", layer_name, n_samples)

            # Select reference subset based on alpha
            n = int(self.alpha * n_samples) if self.alpha < 1.0 else n_samples
            n = max(n, 1)

            if self.alpha < 1.0:
                perm = torch.randperm(n_samples)
                idx = perm[:n]
                self.ref[layer_name] = f[idx]
                self.index[layer_name] = idx
                logger.info("Selected %d samples (alpha=%.3f)", idx.shape[0], self.alpha)
            else:
                self.ref[layer_name] = f
                self.index[layer_name] = torch.arange(n_samples)

            # L2-normalize reference
            self.ref[layer_name] = self.ref[layer_name] / torch.norm(
                self.ref[layer_name], dim=1, keepdim=True
            ).clamp_min(1e-12)

            # Build positions in the sorted lists with the actual 'n'
            kpos = np.array((self.kval * n - 1).astype(int))
            pos_mask = np.concatenate(([True], kpos[:-1] < kpos[1:]))
            self.kval = self.kval[pos_mask]
            self.kpos = kpos[pos_mask]
            self.klen = len(self.kpos)

            # Precompute kNN distance thresholds [n, klen]
            self.knndst = torch.zeros((n, self.klen))
            for i in tqdm(range(n), disable=True):  # disable tqdm in CI
                dst = torch.sum((self.ref[layer_name][i] - self.ref[layer_name]) ** 2, dim=1)  # squared L2
                dst = dst.sort()[0]
                self.knndst[i] = dst[self.kpos + 1]  # +1 skip self-distance

    @torch.no_grad()
    def _layer_score(
        self,
        features: Tensor,
        layer_name: Optional[str] = None,
        index: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute negative k-values as anomaly scores for the given features.
        """
        # Bring everything to the detector's device
        device = self.device
        self.ref[layer_name] = self.ref[layer_name].to(device)
        self.knndst = self.knndst.to(device)

        # Flatten + L2-normalize input features
        x = features
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        x = x.to(device)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)

        n = len(self.ref[layer_name])
        scores = []

        for feature in x:
            dst = torch.sum((feature - self.ref[layer_name]) ** 2, dim=1)
            frac = (self.knndst >= dst.view(n, 1)).sum(dim=0).float() / n

            kval = 1.0  # default score (most conformant)
            for i in range(self.klen):
                if frac[i] >= self.k:
                    kval = float(self.kval[i])
                    break

            scores.append(-kval)  # negative k-value as anomaly score

        return torch.tensor(scores, device=device)
