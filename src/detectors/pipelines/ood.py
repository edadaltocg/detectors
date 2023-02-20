import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import accelerate
import numpy as np
import torch
import torch.utils.data
import torchmetrics.functional as metrics
from torch import Tensor
from tqdm import tqdm

from detectors.data import create_dataset
from detectors.methods.ood import OODDetector
from detectors.pipelines import register_pipeline
from detectors.pipelines.base import Pipeline
from detectors.utils import ConcatDatasetsDim1

_logger = logging.getLogger(__name__)


def fpr_at_fixed_tpr(fprs, tprs, thresholds, tpr_level: float = 0.95):
    # return np.interp(tpr_level, tprs, fprs)
    if all(tprs < tpr_level):
        raise ValueError(f"No threshold allows for TPR at least {tpr_level}.")
    idxs = [i for i, x in enumerate(tprs) if x >= tpr_level]
    idx = min(idxs)
    return fprs[idx], tprs[idx], thresholds[idx]


def compute_detection_error(op_fpr, op_tpr, pos_ratio):
    """Return the misclassification probability when TPR is fixed."""
    # Get ratios of positives to negatives
    neg_ratio = 1 - pos_ratio
    # Get indexes of all TPR >= fixed tpr level
    detection_error = pos_ratio * (1 - op_tpr) + neg_ratio * op_fpr
    return detection_error


METRICS_NAMES_PRETTY = {
    "fpr_at_0.95_tpr": "FPR at 95% TPR",
    "tnr_at_0.95_tpr": "TNR at 95% TPR",
    "detection_error": "Detection error",
    "auroc": "AUROC",
    "aupr_in": "AUPR in",
    "aupr_out": "AUPR out",
    "thr": "Threshold",
    "time": "Time",
}


# TODO: impelement validation of hyperparams
class OODBenchmarkPipeline(Pipeline, ABC):
    def __init__(
        self,
        in_dataset_name: str,
        ood_datasets_names_splits: Dict[str, Any],
        limit_fit: int,
        transform: Optional[Callable],
        batch_size: int,
        seed: int = 42,
    ) -> None:
        self.in_dataset = in_dataset_name
        self.ood_datasets_names_splits = ood_datasets_names_splits
        self.ood_datasets = list(ood_datasets_names_splits.keys())
        self.ood_splits = list(ood_datasets_names_splits.values())
        self.limit_fit = limit_fit
        self.transform = transform
        self.batch_size = batch_size

        self.in_dist_train_dataset = None
        self.in_dist_test_dataset = None
        self.out_distribution_datasets = None
        self.fit_dataloader = None
        self.test_dataloader = None
        self.method = None

        accelerate.utils.set_seed(seed)
        self.accelerator = accelerate.Accelerator()
        super().__init__()

    @abstractmethod
    def _setup(self):
        ...

    def _setup_dataloaders(self):
        if (
            self.in_dist_train_dataset is None
            or self.in_dist_test_dataset is None
            or self.out_distribution_datasets is None
        ):
            raise ValueError("Data loaders are not set.")

        if self.limit_fit is None:
            self.limit_fit = len(self.in_dist_train_dataset)

        # random indices
        subset = np.random.choice(np.arange(len(self.in_dist_train_dataset)), self.limit_fit, replace=False).tolist()
        self.in_dist_train_dataset = torch.utils.data.Subset(self.in_dist_train_dataset, subset)
        _logger.info(f"Using {len(self.in_dist_train_dataset)} samples for fitting.")
        self.fit_dataloader = torch.utils.data.DataLoader(
            self.in_dist_train_dataset, batch_size=self.batch_size, shuffle=True
        )

        test_images_dataset = torch.utils.data.ConcatDataset(
            [self.in_dist_test_dataset] + list(self.out_distribution_datasets.values())
        )
        _logger.info(f"Using {len(test_images_dataset)} samples for testing.")
        test_labels = torch.utils.data.TensorDataset(
            torch.cat(
                [torch.zeros(len(self.in_dist_test_dataset))]  # type: ignore
                + [torch.ones(len(d)) * (i + 1) for i, d in enumerate(self.out_distribution_datasets.values())]  # type: ignore
            ).long()
        )

        test_dataset = ConcatDatasetsDim1([test_images_dataset, test_labels])
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        self.fit_dataloader = self.accelerator.prepare(self.fit_dataloader)
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

    def setup(self):
        self._setup()
        self._setup_dataloaders()

    def preprocess(self, method: OODDetector) -> OODDetector:
        if self.fit_dataloader is None:
            raise ValueError("Fit dataloader is not set.")
        progress_bar = tqdm(
            range(len(self.fit_dataloader)), desc="Fitting", disable=not self.accelerator.is_local_main_process
        )
        method.start()
        for x, y in self.fit_dataloader:
            method.update(x, y)
            progress_bar.update(1)
        self.accelerator.wait_for_everyone()
        method.end()
        self.method = method
        return method

    def forward(self, x: Tensor, *args, **kwargs):
        if self.method is None:
            raise ValueError("Method is not set.")
        return self.method(x)

    def benchmark(self, method: OODDetector):
        self.method = method
        if hasattr(method.detector, "update"):
            self.preprocess(method)
        self.infer_times = []
        test_labels = []
        test_scores = []
        progress_bar = tqdm(
            range(len(self.test_dataloader)), desc="Inference", disable=not self.accelerator.is_local_main_process
        )
        for x, y, labels in self.test_dataloader:
            test_labels.append(labels.cpu())
            t1 = time.time()
            s = self.forward(x)
            t2 = time.time()
            test_scores.append(s.detach().cpu())

            self.infer_times.append(t2 - t1)
            progress_bar.update(1)
        self.accelerator.wait_for_everyone()
        self.infer_times = np.mean(self.infer_times)
        test_labels = torch.cat(test_labels).view(-1)
        test_scores = torch.cat(test_scores).view(-1)
        _logger.info("Computing metrics...")
        res_obj = self.postprocess(test_scores, test_labels, self.ood_datasets)

        return res_obj

    def postprocess(self, test_scores: Tensor, test_labels: Tensor, ood_datasets: List[str]):
        in_scores = test_scores[test_labels == 0]
        in_labels = torch.ones(len(test_labels[test_labels == 0]))

        results = {}
        for i, ood_dataset in enumerate(ood_datasets):
            ood_scores = test_scores[test_labels == (i + 1)]
            ood_labels = torch.zeros(len(test_labels[test_labels == (i + 1)]))

            _test_scores = torch.cat([in_scores, ood_scores])
            _test_labels = torch.cat([in_labels, ood_labels])

            fprs, tprs, thrs = metrics.roc(_test_scores, _test_labels)
            precision, recall, _ = metrics.precision_recall_curve(_test_scores, _test_labels)
            precision_out, recall_out, _ = metrics.precision_recall_curve(_test_scores, _test_labels, pos_label=0)
            fpr, tpr, thr = fpr_at_fixed_tpr(fprs, tprs, thrs, 0.95)
            fpr, tpr, thr = fpr.item(), tpr.item(), thr.item()
            auroc = metrics.auc(fprs, tprs).item()
            aupr_in = metrics.auc(recall, precision, reorder=True).item()
            aupr_out = metrics.auc(recall_out, precision_out, reorder=True).item()

            pos_ratio = torch.mean((_test_labels == 1).float()).item()
            detection_error = compute_detection_error(fpr, tpr, pos_ratio)

            results[ood_dataset] = {
                "fpr_at_0.95_tpr": fpr,
                "tnr_at_0.95_tpr": 1 - fpr,
                "detection_error": detection_error,
                "auroc": auroc,
                "aupr_in": aupr_in,
                "aupr_out": aupr_out,
                "thr": thr,
            }

        results["average"] = {
            "fpr_at_0.95_tpr": np.mean([results[ds]["fpr_at_0.95_tpr"] for ds in ood_datasets]),
            "tnr_at_0.95_tpr": np.mean([results[ds]["tnr_at_0.95_tpr"] for ds in ood_datasets]),
            "detection_error": np.mean([results[ds]["detection_error"] for ds in ood_datasets]),
            "auroc": np.mean([results[ds]["auroc"] for ds in ood_datasets]),
            "aupr_in": np.mean([results[ds]["aupr_in"] for ds in ood_datasets]),
            "aupr_out": np.mean([results[ds]["aupr_out"] for ds in ood_datasets]),
            "thr": np.mean([results[ds]["thr"] for ds in ood_datasets]),
            "time": self.infer_times,
        }

        return results

    @classmethod
    def report(cls, results: Dict[str, Dict[str, Any]]):
        # log results
        print("OOD results:")
        for ood_dataset, res in results.items():
            print(f"\t{ood_dataset}:")
            for metric, val in res.items():
                print(f"\t\t{METRICS_NAMES_PRETTY.get(metric, metric)}: {val:.4f}")


@register_pipeline("ood_cifar10")
class OODCifar10Pipeline(OODBenchmarkPipeline):
    def __init__(self, transform: Any, limit_fit=10000, batch_size=128) -> None:
        super().__init__(
            "cifar10",
            {
                "cifar100": "test",
                "svhn": "test",
                "isun": None,
                "lsun_c": None,
                "lsun_r": None,
                "tiny_imagenet_c": None,
                "tiny_imagenet_r": None,
                "textures": None,
                # "places365": "val",
                "english_chars": None,
            },
            limit_fit=limit_fit,
            transform=transform,
            batch_size=batch_size,
        )

    def _setup(self):
        _logger.info("Loading In-distribution dataset...")
        self.in_dist_train_dataset = create_dataset(
            self.in_dataset, split="train", transform=self.transform, download=True
        )
        self.in_dist_test_dataset = create_dataset(
            self.in_dataset, split="test", transform=self.transform, download=True
        )

        _logger.info("Loading OOD datasets...")
        self.out_distribution_datasets = {
            ds: create_dataset(ds, split=split, transform=self.transform, download=True)
            for ds, split in self.ood_datasets_names_splits.items()
        }


@register_pipeline("ood_cifar100")
class OODCifar100Pipeline(OODCifar10Pipeline):
    def __init__(self, transform: Callable, limit_fit=10000, batch_size=128) -> None:
        super().__init__(
            transform=transform,
            limit_fit=limit_fit,
            batch_size=batch_size,
        )
        self.in_dataset = "cifar100"
        self.ood_datasets_names_splits = {
            "cifar100": "test",
            "svhn": "test",
            "isun": None,
            "lsun_c": None,
            "lsun_r": None,
            "tiny_imagenet_c": None,
            "tiny_imagenet_r": None,
            "textures": None,
            # "places365": "val",
            "english_chars": None,
        }


@register_pipeline("ood_imagenet")
class OODImageNettPipeline(OODBenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=100000, batch_size=64) -> None:
        super().__init__(
            "ilsvrc2012",
            {
                "mos_inaturalist": None,
                "mos_sun": None,
                "mos_places365": None,
                "textures": None,
                "imagenet_o": None,
            },
            limit_fit=limit_fit,
            transform=transform,
            batch_size=batch_size,
        )

    def _setup(self):
        _logger.info("Loading In-distribution dataset...")
        self.in_dist_train_dataset = create_dataset(self.in_dataset, split="train", transform=self.transform)
        self.in_dist_test_dataset = create_dataset(self.in_dataset, split="val", transform=self.transform)

        _logger.info("Loading OOD datasets...")
        self.out_distribution_datasets = {
            ds: create_dataset(ds, split="test", transform=self.transform, download=True) for ds in self.ood_datasets
        }
