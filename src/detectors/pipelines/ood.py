import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import accelerate
import numpy as np
import optuna
import pandas as pd
import sklearn.metrics
import torch
import torch.utils.data
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
        self.setup()

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

            # self.fit_dataloader = self.accelerator.prepare(self.fit_dataloader) # this can cause bugs
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

    def preprocess(self, method: OODDetector) -> OODDetector:
        if self.fit_dataset is None:
            _logger.warning("Fit is not set or not supported. Returning.")
            return method

        if method.model is not None:
            method.detector.model = self.accelerator.prepare(self.method.detector.model)

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

    def _run(self):
        if hasattr(self.method.detector, "update"):
            self.preprocess(self.method)
        self.infer_times = []
        test_labels = []
        test_scores = []
        progress_bar = tqdm(
            range(len(self.test_dataloader)), desc="Inference", disable=not self.accelerator.is_local_main_process
        )
        for x, y, labels in self.test_dataloader:
            t1 = time.time()
            score = self.method(x)
            t2 = time.time()

            labels, score = self.accelerator.gather_for_metrics((labels, score))

            test_labels.append(labels.cpu())
            test_scores.append(score.detach().cpu())
            self.infer_times.append(t2 - t1)

            progress_bar.update(1)
        self.accelerator.wait_for_everyone()
        self.infer_times = np.mean(self.infer_times)
        test_scores = torch.cat(test_scores).view(-1)
        test_labels = torch.cat(test_labels).view(-1)
        return test_scores, test_labels

    def run(self, method: OODDetector) -> Dict[str, Any]:
        self.method = method

        if self.method.model is not None:
            _logger.info("Preparing model...")
            self.method.model = self.accelerator.prepare(self.method.model)

        _logger.info("Running pipeline...")
        test_scores, test_labels = self._run()
        in_scores = test_scores[test_labels == 0]
        ood_scores = test_scores[test_labels == 1]

        _logger.info("Computing metrics...")
        res_obj = self.postprocess(in_scores, ood_scores)

        return res_obj

    def postprocess(self, in_scores: Tensor, ood_scores: Tensor) -> Dict[str, float]:
        in_labels = torch.ones(len(in_scores))
        ood_labels = torch.zeros(len(ood_scores))

        _test_scores = torch.cat([in_scores, ood_scores]).cpu().numpy()
        _test_labels = torch.cat([in_labels, ood_labels]).cpu().numpy()

        fprs, tprs, thrs = sklearn.metrics.roc_curve(_test_labels, _test_scores)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(_test_labels, _test_scores, pos_label=1)
        precision_out, recall_out, _ = sklearn.metrics.precision_recall_curve(_test_labels, _test_scores, pos_label=0)
        fpr, tpr, thr = fpr_at_fixed_tpr(fprs, tprs, thrs, 0.95)
        auroc = sklearn.metrics.auc(fprs, tprs)
        aupr_in = sklearn.metrics.auc(recall, precision)
        aupr_out = sklearn.metrics.auc(recall_out, precision_out)

        pos_ratio = np.mean(_test_labels == 1)
        detection_error = compute_detection_error(fpr, tpr, pos_ratio)

        results = {
            "fpr_at_0.95_tpr": fpr,
            "tnr_at_0.95_tpr": 1 - fpr,
            "detection_error": detection_error,
            "auroc": auroc,
            "aupr_in": aupr_in,
            "aupr_out": aupr_out,
            "thr": thr,
            "time": self.infer_times,
        }
        return results


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
        self.limit_fit = limit_fit
        self.transform = transform
        self.batch_size = batch_size
        self.seed = seed

        self.fit_dataset = None
        self.in_dataset = None
        self.out_dataset = None
        self.out_datasets = None
        self.val_dataset = None
        self.fit_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.method = None

        self.accelerator = accelerate.Accelerator()
        accelerate.utils.set_seed(seed)
        self.setup()

    @abstractmethod
    def _setup(self):
        """Setup `in_dataset`, `out_dataset`, `fit_dataset` and `out_datasets`."""
        ...

    def _setup_dataloaders(self):
        if self.fit_dataset is None or self.in_dataset is None or self.out_datasets is None:
            raise ValueError("Datasets are not set.")

        if self.limit_fit is None:
            self.limit_fit = len(self.fit_dataset)

        # random indices
        subset = np.random.choice(np.arange(len(self.fit_dataset)), self.limit_fit, replace=False).tolist()
        self.fit_dataset = torch.utils.data.Subset(self.fit_dataset, subset)
        self.fit_dataloader = torch.utils.data.DataLoader(self.fit_dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = torch.utils.data.ConcatDataset([self.in_dataset, self.out_dataset])
        test_labels = torch.utils.data.TensorDataset(
            torch.cat(
                [torch.zeros(len(self.in_dataset))]  # type: ignore
                + [torch.ones(len(d)) * (i + 1) for i, d in enumerate(self.out_datasets.values())]  # type: ignore
            ).long()
        )

        test_dataset = ConcatDatasetsDim1([test_dataset, test_labels])
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        self.fit_dataloader = self.accelerator.prepare(self.fit_dataloader)
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

        _logger.info(f"Using {len(self.fit_dataset)} samples for fitting.")
        _logger.info(f"Using {len(test_dataset)} samples for testing.")

    def setup(self):
        self._setup()
        self._setup_dataloaders()

    def preprocess(self, method: OODDetector) -> OODDetector:
        return super().preprocess(method)

    def run(self, method: OODDetector) -> Dict[str, Any]:
        self.method = method

        if self.method.model is not None:
            _logger.info("Preparing model...")
            self.method.model = self.accelerator.prepare(self.method.model)

        _logger.info("Running pipeline...")
        test_scores, test_labels = self._run()
        res_obj = self.postprocess(test_scores, test_labels)

        return res_obj

    def postprocess(self, test_scores: Tensor, test_labels: Tensor):
        _logger.info("Computing metrics...")
        in_scores = test_scores[test_labels == 0]

        results = {}
        for i, ood_dataset_name in enumerate(self.out_datasets_names):
            ood_scores = test_scores[test_labels == (i + 1)]
            results[ood_dataset_name] = super().postprocess(in_scores, ood_scores)

        results["average"] = {
            k: np.mean([results[ds][k] for ds in self.out_datasets_names])
            for k in results[self.out_datasets_names[0]].keys()
        }

        return results

    def report(self, results: Dict[str, Dict[str, Any]]) -> str:
        # log results in a table
        df = pd.DataFrame()

        for ood_dataset, res in results.items():
            df = pd.concat([df, pd.DataFrame(res, index=[ood_dataset])])
            # df = df.append(
            #     {"Dataset": ood_dataset, **{METRICS_NAMES_PRETTY[k]: v for k, v in res.items()}}, ignore_index=True
            # )
        df.columns = [METRICS_NAMES_PRETTY[k] for k in df.columns]
        return df.to_string(index=False)


@register_pipeline("ood_cifar10_benchmark")
class OODCifar10BenchmarkPipeline(OODBenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=10000, batch_size=128, seed=42, **kwargs) -> None:
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
                "places365": None,
                "english_chars": None,
            },
            transform=transform,
            batch_size=batch_size,
            limit_fit=limit_fit,
            seed=seed,
        )

    def _setup(self):
        _logger.info("Loading In-distribution dataset...")
        self.fit_dataset = create_dataset(self.in_dataset_name, split="train", transform=self.transform, download=True)
        self.in_dataset = create_dataset(self.in_dataset_name, split="test", transform=self.transform, download=True)

        _logger.info("Loading OOD datasets...")
        self.out_datasets = {
            ds: create_dataset(ds, split=split, transform=self.transform, download=True)
            for ds, split in self.out_datasets_names_splits.items()
        }
        self.out_dataset = torch.utils.data.ConcatDataset(list(self.out_datasets.values()))


@register_pipeline("ood_cifar100_benchmark")
class OODCifar100BenchmarkPipeline(OODCifar10BenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=10000, batch_size=128, seed=42, **kwargs) -> None:
        super().__init__(transform=transform, limit_fit=limit_fit, batch_size=batch_size, seed=seed)
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
            "places365": None,
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
        self.fit_dataset = create_dataset(self.in_dataset_name, split="train", transform=self.transform)
        self.in_dataset = create_dataset(self.in_dataset_name, split="val", transform=self.transform)

        _logger.info("Loading OOD datasets...")
        self.out_datasets = {
            ds: create_dataset(ds, split=split, transform=self.transform, download=True)
            for ds, split in self.out_datasets_names_splits.items()
        }
        self.out_dataset = torch.utils.data.ConcatDataset(list(self.out_datasets.values()))


@register_pipeline("ood_mnist_benchmark")
class OODMNISTBenchmarkPipeline(OODBenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=100000, batch_size=64) -> None:
        super().__init__(
            "mnist",
            {
                "fashion_mnist": "test",
                "svhn": "test",
                "cifar10": "test",
                "textures": None,
                "english_chars": None,
            },
            limit_fit=limit_fit,
            transform=transform,
            batch_size=batch_size,
        )

    def _setup(self):
        _logger.info("Loading In-distribution dataset...")
        self.fit_dataset = create_dataset(self.in_dataset_name, split="train", transform=self.transform)
        self.in_test_dataset = create_dataset(self.in_dataset_name, split="test", transform=self.transform)

        _logger.info("Loading OOD datasets...")
        self.out_datasets = {
            ds: create_dataset(ds, split=split, transform=self.transform, download=True)
            for ds, split in self.out_datasets_names_splits.items()
        }
        self.out_dataset = torch.utils.data.ConcatDataset(list(self.out_datasets.values()))


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

    def report(self, study: optuna.study.Study):
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
