import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import accelerate
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.utils.data
from torch import Tensor
from tqdm import tqdm

from detectors.data import create_dataset
from detectors.eval import METRICS_NAMES_PRETTY, get_ood_results
from detectors.methods import DetectorWrapper
from detectors.methods.templates import Detector
from detectors.pipelines import register_pipeline
from detectors.pipelines.base import Pipeline
from detectors.utils import ConcatDatasetsDim1
from sklearn import metrics

_logger = logging.getLogger(__name__)


def mmd_rbf(X, Y, gamma=1.0, XX=None):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    if XX is None:
        XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


class DriftPipeline(Pipeline, ABC):
    _sign = -1

    def __init__(
        self,
        in_dataset_name: str,
        drift_datasets_names_splits: Dict[str, Any],
        transform,
        window_sizes: List[int] = [10, 20, 50, 100, 200, 500, 1000],
        ood_ratio: float = 1.0,
        num_samples: int = 10000,
        warmup_size=10000,
        disable_pred=False,
        batch_size=128,
        limit_fit: float = 1.0,
        num_workers: int = 8,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        seed=42,
    ) -> None:
        self.window_sizes = window_sizes
        self.num_samples = num_samples
        self.in_dataset_name = in_dataset_name
        self.drift_datasets_names_splits = drift_datasets_names_splits
        self.seed = seed
        self.warmup_size = warmup_size
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.limit_fit = limit_fit
        self.disable_pred = disable_pred
        self.mix_factor = ood_ratio
        assert 0 <= self.mix_factor <= 1, "Mix factor must be in [0, 1]."

        self.drift_datasets_names = list(self.drift_datasets_names_splits.keys())

        self.in_dataset = None
        self.fit_dataset = None
        self.warmup_dataset = None
        self.drift_dataset = None

        self.in_dataloader = None
        self.fit_dataloader = None
        self.warmup_dataloader = None
        self.drift_dataloader = None

        self.warmup_scores = None
        self.in_scores = None
        self.drift_scores = None

        accelerate.utils.set_seed(self.seed)

    @abstractmethod
    def _setup_datasets(self):
        """Setup `in_dataset`, `drift_dataset`, and `fit_dataset`."""
        ...

    def _setup_dataloaders(self):
        if self.fit_dataset is None or self.in_dataset is None or self.drift_dataset is None:
            raise ValueError("Datasets are not set.")
        self.limit_fit = min(int(self.limit_fit * len(self.fit_dataset)), len(self.fit_dataset))

        subset = np.random.choice(np.arange(len(self.fit_dataset)), self.limit_fit, replace=False).tolist()
        self.fit_dataset = torch.utils.data.Subset(self.fit_dataset, subset)
        self.fit_dataloader = torch.utils.data.DataLoader(
            self.fit_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

        self.warmup_dataset = torch.utils.data.Subset(self.fit_dataset, np.arange(self.warmup_size).tolist())
        self.warmup_dataloader = torch.utils.data.DataLoader(
            self.warmup_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

        self.in_dataloader = torch.utils.data.DataLoader(
            self.in_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

        self.drift_dataloader = torch.utils.data.DataLoader(
            self.drift_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

        _logger.info(f"Using {len(self.fit_dataset)} samples for fitting.")
        _logger.info(f"Using {len(self.in_dataset)} samples for in-dsitribution.")
        _logger.info(f"Using {len(self.drift_dataset)} samples for drifted data.")

    def save_pretrained(self, path):
        raise NotImplementedError

    def load_pretrained(self, path):
        raise NotImplementedError

    def setup(self):
        self._setup_datasets()
        self._setup_dataloaders()

    def preprocess(self, method: Union[DetectorWrapper, Detector], *args, **kwargs) -> Union[DetectorWrapper, Detector]:
        if self.fit_dataset is None or self.fit_dataloader is None:
            _logger.warning("Fit dataset is not set or not supported. Returning.")
            return method

        if not hasattr(method.detector, "update"):
            _logger.warning("Detector does not support fitting. Returning.")
            return method

        progress_bar = tqdm(range(len(self.fit_dataloader)), desc="Fitting")
        fit_length = len(self.fit_dataloader.dataset)
        example = next(iter(self.fit_dataloader))[0]
        method.start(example=example, fit_length=fit_length)
        for x, y in self.fit_dataloader:
            method.update(x, y)
            progress_bar.update(1)
        progress_bar.close()
        method.end()
        return method

    def run(self, method: Union[DetectorWrapper, Detector]) -> Dict[str, Any]:
        self.method = method

        _logger.info("Running pipeline...")
        self.method = self.preprocess(self.method)

        sample_idx = 0
        self.warmup_scores = torch.zeros(len(self.warmup_dataset), dtype=torch.float32)
        self.warmup_labels = torch.zeros(len(self.warmup_dataset), dtype=torch.int32)
        self.warmup_preds = torch.zeros(len(self.warmup_dataset), dtype=torch.int32)
        self.infer_times = torch.zeros(len(self.warmup_dataloader), dtype=torch.float32)
        progress_bar = tqdm(range(len(self.warmup_dataloader)), desc="Warmup inference")
        for batch_idx, (x, y) in enumerate(self.warmup_dataloader):
            t1 = time.time()
            score = self.method(x)
            t2 = time.time()
            if not self.disable_pred:
                pred = self.method.model(x.to(self.method.device)).argmax(dim=1)
                self.warmup_preds[sample_idx : sample_idx + len(x)] = pred.cpu()
            else:
                self.warmup_preds = None
            self.warmup_scores[sample_idx : sample_idx + len(x)] = score.cpu()
            self.warmup_labels[sample_idx : sample_idx + len(x)] = y.cpu()
            self.infer_times[batch_idx] = t2 - t1

            sample_idx += len(x)
            progress_bar.update(1)
        progress_bar.close()

        sample_idx = 0
        self.in_scores = torch.zeros(len(self.in_dataset), dtype=torch.float32)
        self.in_labels = torch.zeros(len(self.in_dataset), dtype=torch.int32)
        self.in_preds = torch.zeros(len(self.in_dataset), dtype=torch.int32)
        progress_bar = tqdm(range(len(self.in_dataloader)), desc="In-distribution inference")
        for batch_idx, (x, y) in enumerate(self.in_dataloader):
            score = self.method(x)
            if not self.disable_pred:
                pred = self.method.model(x.to(self.method.device)).argmax(dim=1)
                self.in_preds[sample_idx : sample_idx + len(x)] = pred.cpu()
            else:
                self.in_preds = None
            self.in_scores[sample_idx : sample_idx + len(x)] = score.cpu()
            self.in_labels[sample_idx : sample_idx + len(x)] = y.cpu()

            sample_idx += len(x)
            progress_bar.update(1)
        progress_bar.close()

        sample_idx = 0
        self.drift_scores = torch.zeros(len(self.drift_dataset), dtype=torch.float32)
        self.drift_labels = torch.zeros(len(self.drift_dataset), dtype=torch.int32)
        self.drift_preds = torch.zeros(len(self.drift_dataset), dtype=torch.int32)
        progress_bar = tqdm(range(len(self.drift_dataloader)), desc="Drift inference")
        for batch_idx, (x, y) in enumerate(self.drift_dataloader):
            score = self.method(x)
            if not self.disable_pred:
                pred = self.method.model(x.to(self.method.device)).argmax(dim=1)
                self.drift_preds[sample_idx : sample_idx + len(x)] = pred.cpu()
            else:
                self.drift_preds = None
            self.drift_scores[sample_idx : sample_idx + len(x)] = score.cpu()
            self.drift_labels[sample_idx : sample_idx + len(x)] = y.cpu()

            sample_idx += len(x)
            progress_bar.update(1)
        progress_bar.close()

        return {
            "warmup_scores": self.warmup_scores,
            "warmup_labels": self.warmup_labels,
            "warmup_preds": self.warmup_preds,
            "in_scores": self.in_scores,
            "in_labels": self.in_labels,
            "in_preds": self.in_preds,
            "drift_scores": self.drift_scores,
            "drift_labels": self.drift_labels,
            "drift_preds": self.drift_preds,
            "time": self.infer_times,
        }

    def partition_scores(self, scores: Union[Tensor, np.ndarray]) -> Dict[int, np.ndarray]:
        """Partition scores into windows.

        Args:
            scores (Tensor): Scores to be partitioned.

        Returns:
            List[Tensor]: List of scores for each window.
        """
        if isinstance(scores, Tensor):
            scores = scores.numpy()
        scores_windows = {ws: np.empty((self.num_samples // ws, ws), dtype=np.float32) for ws in self.window_sizes}
        for window_size in self.window_sizes:
            n = len(scores)
            for i in range(len(scores_windows[window_size])):
                indexes = np.random.choice(np.arange(n), window_size, replace=False)
                scores_windows[window_size][i, :] = scores[indexes]

        return scores_windows

    def postprocess(
        self,
        warmup_scores: Tensor,
        in_scores: Tensor,
        drift_scores: Tensor,
        warmup_labels: Optional[Tensor] = None,
        in_labels: Optional[Tensor] = None,
        drift_labels: Optional[Tensor] = None,
        warmup_preds: Optional[Tensor] = None,
        in_preds: Optional[Tensor] = None,
        drift_pred: Optional[Tensor] = None,
        criterion="ks_2samp",
    ):
        n = int(max(self.window_sizes) * 2)  # 2048
        # shuffle in_scores in a reproducible manner
        in_scores = in_scores.numpy().reshape(-1)
        drift_scores = drift_scores.numpy().reshape(-1)
        np.random.shuffle(in_scores)
        np.random.shuffle(drift_scores)

        in_scores_windows = self.partition_scores(in_scores[2 * n :])
        drift_scores_windows = self.partition_scores(drift_scores)

        sample_reference = in_scores[:n]
        sample_ind_val = in_scores[n : 2 * n]
        sample_reference_samples = self.partition_scores(sample_ind_val)

        if criterion == "mmd_rbf":
            XX = metrics.pairwise.rbf_kernel(sample_reference.reshape(-1, 1), sample_reference.reshape(-1, 1))

        results = {}
        _logger.info("Computing metrics...")
        for ws in self.window_sizes:
            in_test_scores = np.empty(len(in_scores_windows[ws]))
            out_test_scores = np.empty(len(in_scores_windows[ws]))
            in_p_values = np.ones(len(in_scores_windows[ws])) * -1
            out_p_values = np.ones(len(in_scores_windows[ws])) * -1
            assert len(in_test_scores) == len(out_test_scores) == len(in_p_values) == len(out_p_values)
            for w_idx in range(len(in_scores_windows[ws])):
                # build mixture distribution
                indexes = np.random.choice(np.arange(ws), int(np.ceil((1 - self.mix_factor) * ws)), replace=False)
                test_mix_in = in_scores_windows[ws][w_idx][indexes]
                indexes = np.random.choice(np.arange(ws), int(np.ceil(self.mix_factor * ws)), replace=False)
                test_mix_out = drift_scores_windows[ws][w_idx][indexes]

                test_mix = np.concatenate([test_mix_in, test_mix_out])
                np.random.shuffle(test_mix)
                test_mix = test_mix[:ws]
                assert len(test_mix) == ws, f"len(test_mix)={len(test_mix)} != ws={ws}"  # sanity test

                if criterion == "ks_2samp":
                    in_test_scores[w_idx], in_p_values[w_idx] = stats.ks_2samp(
                        sample_reference, sample_reference_samples[ws][w_idx]
                    )
                    out_test_scores[w_idx], out_p_values[w_idx] = stats.ks_2samp(sample_reference, test_mix)
                elif criterion == "mmd_rbf":
                    in_test_scores[w_idx] = mmd_rbf(sample_reference, sample_reference_samples[ws][w_idx], XX=XX)
                    out_test_scores[w_idx] = mmd_rbf(sample_reference, test_mix, XX=XX)

            # assert np.isnan(in_test_scores).sum() == 0, np.isnan(in_test_scores).sum()
            # assert np.isnan(out_test_scores).sum() == 0, np.isnan(out_test_scores).sum()
            results[ws] = get_ood_results(self._sign * in_test_scores, self._sign * out_test_scores)  # type: ignore

        return results

    def report(self, results: Dict[str, Dict[str, Any]]) -> str:
        # log results in a table
        if "results" in results:
            results = results["results"]
        df = pd.DataFrame()

        for ws, res in results.items():
            if not isinstance(res, dict):
                continue
            df = pd.concat([df, pd.DataFrame(res, index=[ws])])
        df.columns = [METRICS_NAMES_PRETTY[k] for k in df.columns]
        return df.to_string(index=True, float_format="{:.3f}".format)


@register_pipeline("drift_benchmark_imagenet_r")
class DriftImageNetRBenchmarkPipelined(DriftPipeline):
    def __init__(
        self,
        transform: Callable,
        window_sizes=[10, 20, 50, 100, 200, 500, 1000],
        mix_factor=1.0,
        warmup_size=10000,
        limit_fit=1.0,
        batch_size=64,
        seed=42,
        **kwargs,
    ) -> None:
        super().__init__(
            "ilsvrc2012",
            {
                "imagenet_r": None,
            },
            transform=transform,
            window_sizes=window_sizes,
            ood_ratio=mix_factor,
            warmup_size=warmup_size,
            limit_fit=limit_fit,
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )

    def _setup_datasets(self):
        _logger.info("Loading In-distribution dataset...")
        self.fit_dataset = create_dataset(self.in_dataset_name, split="train", transform=self.transform)
        self.in_dataset = create_dataset(self.in_dataset_name, split="val", transform=self.transform)

        _logger.info("Loading OOD datasets...")
        drift_datasets = {
            ds: create_dataset(ds, split=split, transform=self.transform, download=True)
            for ds, split in self.drift_datasets_names_splits.items()
        }
        self.drift_dataset = torch.utils.data.ConcatDataset(list(drift_datasets.values()))


@register_pipeline("drift_benchmark_imagenet_lt")
class DriftImageNetLTBenchmarkPipelined(DriftPipeline):
    _sign = -1

    def __init__(
        self,
        transform: Callable,
        window_sizes=[10, 20, 50, 100, 200, 500, 1000],
        mix_factor=1.0,
        warmup_size=10000,
        limit_fit=1.0,
        batch_size=64,
        seed=42,
        **kwargs,
    ) -> None:
        super().__init__(
            "ilsvrc2012",
            {
                "imagenet_lt": "val",
            },
            transform=transform,
            window_sizes=window_sizes,
            ood_ratio=mix_factor,
            warmup_size=warmup_size,
            limit_fit=limit_fit,
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )

    def _setup_datasets(self):
        _logger.info("Loading In-distribution dataset...")
        self.fit_dataset = create_dataset(self.in_dataset_name, split="train", transform=self.transform)
        self.in_dataset = create_dataset(self.in_dataset_name, split="val", transform=self.transform)

        _logger.info("Loading OOD datasets...")
        drift_datasets = {
            ds: create_dataset(ds, split=split, transform=self.transform)
            for ds, split in self.drift_datasets_names_splits.items()
        }
        self.drift_dataset = torch.utils.data.ConcatDataset(list(drift_datasets.values()))


@register_pipeline("drift_benchmark_openimage_o")
class DriftOpenImageOBenchmarkPipelined(DriftPipeline):
    _sign = -1

    def __init__(
        self,
        transform: Callable,
        window_sizes=[10, 20, 50, 100, 200, 500, 1000],
        mix_factor=1.0,
        warmup_size=10000,
        limit_fit=1.0,
        batch_size=64,
        seed=42,
        **kwargs,
    ) -> None:
        super().__init__(
            "ilsvrc2012",
            {
                "openimage_o": None,
            },
            transform=transform,
            window_sizes=window_sizes,
            ood_ratio=mix_factor,
            warmup_size=warmup_size,
            limit_fit=limit_fit,
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )

    def _setup_datasets(self):
        _logger.info("Loading In-distribution dataset...")
        self.fit_dataset = create_dataset(self.in_dataset_name, split="train", transform=self.transform)
        self.in_dataset = create_dataset(self.in_dataset_name, split="val", transform=self.transform)

        _logger.info("Loading OOD datasets...")
        drift_datasets = {
            ds: create_dataset(ds, split=split, transform=self.transform)
            for ds, split in self.drift_datasets_names_splits.items()
        }
        self.drift_dataset = torch.utils.data.ConcatDataset(list(drift_datasets.values()))
