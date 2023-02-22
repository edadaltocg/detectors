import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import accelerate
import numpy as np
import optuna
import sklearn.metrics
import torch
import torch.utils.data
import torchmetrics.functional as metrics
from optuna.trial import TrialState
from torch import Tensor
from tqdm import tqdm

from detectors.data import create_dataset
from detectors.eval import compute_detection_error, fpr_at_fixed_tpr
from detectors.methods.ood import OODDetector
from detectors.pipelines import register_pipeline
from detectors.pipelines.base import Pipeline
from detectors.utils import ConcatDatasetsDim1

_logger = logging.getLogger(__name__)


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


class OODBasePipeline(Pipeline):
    def __init__(
        self,
        in_dataset: torch.utils.data.Dataset,
        out_dataset: torch.utils.data.Dataset,
        fit_dataset: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 128,
        seed: int = 42,
    ) -> None:
        self.in_dataset = in_dataset
        self.out_dataset = out_dataset
        self.fit_dataset = fit_dataset
        self.batch_size = batch_size

        accelerate.utils.set_seed(seed)
        self.accelerator = accelerate.Accelerator()
        super().__init__()

    def setup(self):
        test_dataset = torch.utils.data.ConcatDataset([self.in_dataset, self.out_dataset])
        test_labels = torch.utils.data.TensorDataset(
            torch.cat([torch.zeros(len(self.in_dataset))] + [torch.ones(len(self.out_dataset))]).long()  # type: ignore
        )

        self.test_dataset = ConcatDatasetsDim1([test_dataset, test_labels])
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
        if self.fit_dataset is not None:
            self.fit_dataloader = torch.utils.data.DataLoader(
                self.fit_dataset, batch_size=self.batch_size, shuffle=True
            )
            self.fit_dataloader = self.accelerator.prepare(self.fit_dataloader)
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

    def preprocess(self, method: OODDetector) -> OODDetector:
        if self.fit_dataset is None:
            _logger.warning("Fit is not set or not supported. Returning.")
            return method

        progress_bar = tqdm(
            range(len(self.fit_dataloader)), desc="Fitting", disable=not self.accelerator.is_local_main_process
        )
        method.start()
        for x, y in self.fit_dataloader:
            method.update(x, y)
            progress_bar.update(1)
        self.accelerator.wait_for_everyone()
        method.end()
        return method

    def _run(self, method: OODDetector):
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
            s = self.method(x)
            t2 = time.time()
            test_scores.append(s.detach().cpu())

            self.infer_times.append(t2 - t1)
            progress_bar.update(1)
        self.accelerator.wait_for_everyone()
        self.infer_times = np.mean(self.infer_times)
        test_labels = torch.cat(test_labels).view(-1)
        test_scores = torch.cat(test_scores).view(-1)
        return test_scores, test_labels

    def run(self, method: OODDetector) -> Tuple[OODDetector, Dict[str, Any]]:
        test_scores, test_labels = self._run(method)
        in_scores = test_scores[test_labels == 0]
        ood_scores = test_scores[test_labels == 1]

        _logger.info("Computing metrics...")
        res_obj = self.postprocess(in_scores, ood_scores)

        return self.method, res_obj

    def postprocess(self, in_scores: Tensor, ood_scores: Tensor):
        in_labels = torch.ones(len(in_scores))
        ood_labels = torch.zeros(len(ood_scores))

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

        results = {
            "fpr_at_0.95_tpr": fpr,
            "tnr_at_0.95_tpr": 1 - fpr,
            "detection_error": detection_error,
            "auroc": auroc,
            "aupr_in": aupr_in,
            "aupr_out": aupr_out,
            "thr": thr,
        }
        return results

    @classmethod
    def report(cls, results: Dict[str, Dict[str, Any]]):
        # log results
        print("OOD results:")
        for metric, val in results.items():
            print(f"\t{METRICS_NAMES_PRETTY.get(metric, metric)}: {val:.4f}")


class OODBenchmarkPipeline(OODBasePipeline, ABC):
    def __init__(
        self,
        in_dataset_name: str,
        out_datasets_names_splits: Dict[str, Any],
        transform: Callable,
        batch_size: int,
        limit_fit: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.in_dataset_name = in_dataset_name
        self.out_datasets_names_splits = out_datasets_names_splits
        self.out_datasets_names = list(out_datasets_names_splits.keys())
        self.ood_datasets_splits = list(out_datasets_names_splits.values())
        self.limit_fit = limit_fit
        self.transform = transform
        self.batch_size = batch_size

        self.in_train_dataset = None
        self.in_test_dataset = None
        self.out_datasets = None
        self.val_dataset = None
        self.fit_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.method = None

        self.setup()

    @abstractmethod
    def _setup(self):
        ...

    def _setup_dataloaders(self):
        if self.in_train_dataset is None or self.in_test_dataset is None or self.out_datasets is None:
            raise ValueError("Datasets are not set.")

        if self.limit_fit is None:
            self.limit_fit = len(self.in_train_dataset)

        # random indices
        subset = np.random.choice(np.arange(len(self.in_train_dataset)), self.limit_fit, replace=False).tolist()
        self.in_train_dataset = torch.utils.data.Subset(self.in_train_dataset, subset)
        _logger.info(f"Using {len(self.in_train_dataset)} samples for fitting.")
        self.fit_dataloader = torch.utils.data.DataLoader(
            self.in_train_dataset, batch_size=self.batch_size, shuffle=True
        )

        test_dataset = torch.utils.data.ConcatDataset([self.in_test_dataset] + list(self.out_datasets.values()))
        _logger.info(f"Using {len(test_dataset)} samples for testing.")
        test_labels = torch.utils.data.TensorDataset(
            torch.cat(
                [torch.zeros(len(self.in_test_dataset))]  # type: ignore
                + [torch.ones(len(d)) * (i + 1) for i, d in enumerate(self.out_datasets.values())]  # type: ignore
            ).long()
        )

        test_dataset = ConcatDatasetsDim1([test_dataset, test_labels])
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        self.fit_dataloader = self.accelerator.prepare(self.fit_dataloader)
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

    def setup(self):
        self._setup()
        self._setup_dataloaders()

    def run(self, method: OODDetector) -> Tuple[OODDetector, Dict[str, Dict[str, Any]]]:
        self.method = method
        test_scores, test_labels = self._run(self.method)
        results = self.postprocess(test_scores, test_labels, self.out_datasets_names)
        return self.method, results

    def postprocess(self, test_scores: Tensor, test_labels: Tensor, out_datasets_names: List[str]):
        in_scores = test_scores[test_labels == 0]

        results = {}
        for i, ood_dataset_name in enumerate(out_datasets_names):
            ood_scores = test_scores[test_labels == (i + 1)]
            results[ood_dataset_name] = super().postprocess(in_scores, ood_scores)

        results["average"] = {
            "fpr_at_0.95_tpr": np.mean([results[ds]["fpr_at_0.95_tpr"] for ds in out_datasets_names]),
            "tnr_at_0.95_tpr": np.mean([results[ds]["tnr_at_0.95_tpr"] for ds in out_datasets_names]),
            "detection_error": np.mean([results[ds]["detection_error"] for ds in out_datasets_names]),
            "auroc": np.mean([results[ds]["auroc"] for ds in out_datasets_names]),
            "aupr_in": np.mean([results[ds]["aupr_in"] for ds in out_datasets_names]),
            "aupr_out": np.mean([results[ds]["aupr_out"] for ds in out_datasets_names]),
            "thr": np.mean([results[ds]["thr"] for ds in out_datasets_names]),
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


@register_pipeline("ood_cifar10_benchmark")
class OODCifar10BenchmarkPipeline(OODBenchmarkPipeline):
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
        self.in_train_dataset = create_dataset(
            self.in_dataset_name, split="train", transform=self.transform, download=True
        )
        self.in_test_dataset = create_dataset(
            self.in_dataset_name, split="test", transform=self.transform, download=True
        )

        _logger.info("Loading OOD datasets...")
        self.out_datasets = {
            ds: create_dataset(ds, split=split, transform=self.transform, download=True)
            for ds, split in self.out_datasets_names_splits.items()
        }


@register_pipeline("ood_cifar100_benchmark")
class OODCifar100BenchmarkPipeline(OODCifar10BenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=10000, batch_size=128) -> None:
        super().__init__(transform=transform, limit_fit=limit_fit, batch_size=batch_size)
        self.in_dataset_name = "cifar100"
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


@register_pipeline("ood_imagenet_benchmark")
class OODImageNetBenchmarkPipeline(OODBenchmarkPipeline):
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
        self.in_train_dataset = create_dataset(self.in_dataset_name, split="train", transform=self.transform)
        self.in_test_dataset = create_dataset(self.in_dataset_name, split="val", transform=self.transform)

        _logger.info("Loading OOD datasets...")
        self.out_datasets = {
            ds: create_dataset(ds, split="test", transform=self.transform, download=True) for ds in self.out_datasets
        }


class OODValidationPipeline(OODBasePipeline):
    def run(
        self,
        method: OODDetector,
        hyperparameters: Dict[str, List[Any]],
        objective_metric: Literal["fpr", "auc"] = "fpr",
        n_trials=100,
    ) -> Tuple[OODDetector, optuna.study.Study]:
        self.method = method
        self.hyperparameters = hyperparameters
        self.objective_metric = objective_metric

        direction = "minimize" if objective_metric == "fpr" else "maximize"
        study = optuna.create_study(direction=direction)
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        self.method = self.method.set_params(**study.best_params)
        return self.method, study

    def objective(self, trial: optuna.trial.Trial) -> float:
        # build detector from trial params
        new_params = {}
        for k in self.hyperparameters:
            new_params[k] = trial.suggest_categorical(k, self.hyperparameters[k])
        self.method = self.method.set_params(**new_params)

        test_scores, test_labels = self._run(self.method)
        in_scores = test_scores[test_labels == 0]
        ood_scores = test_scores[test_labels == 1]

        _logger.info("Computing metrics...")
        results = self.postprocess(in_scores, ood_scores)

        return results[self.objective_metric]

    @classmethod
    def report(cls, study: optuna.study.Study):
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        best_trial = study.best_trial
        best_value = best_trial.value

        report_str = f"""
Study statistics:
    Number of finished trials: {len(study.trials)}
    Number of pruned trials: {len(pruned_trials)}
    Number of complete trials: {len(complete_trials)}
    Best trial:
        Value: {best_value}
        Params: {best_trial.params}
"""
        return report_str


@register_pipeline("ood_cifar10_validation")
class OODCifar10ValidationPipeline(OODValidationPipeline):
    def __init__(self, out_dataset_name: str, transform: Callable, limit_fit=10000, batch_size=128, seed=42) -> None:
        self.in_dataset_name = "cifar10"
        fit_dataset = create_dataset(self.in_dataset_name, split="train", transform=transform, download=True)
        in_dataset = create_dataset(self.in_dataset_name, split="test", transform=transform, download=True)
        out_dataset = create_dataset(out_dataset_name, split=None, transform=transform, download=True)

        super().__init__(
            out_dataset=out_dataset, in_dataset=in_dataset, fit_dataset=fit_dataset, batch_size=batch_size, seed=seed
        )
        subset = np.random.choice(np.arange(len(fit_dataset)), limit_fit, replace=False).tolist()
        self.fit_dataset = torch.utils.data.Subset(self.fit_dataset, subset)


@register_pipeline("ood_cifar100_validation")
class OODCifar100ValidationPipeline(OODCifar10ValidationPipeline):
    def __init__(self, out_dataset_name: str, transform: Callable, limit_fit=10000, batch_size=128, seed=42) -> None:
        self.in_dataset_name = "cifar100"
        super().__init__(out_dataset_name, transform, limit_fit, batch_size, seed)
