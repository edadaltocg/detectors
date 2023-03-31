import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import accelerate
import numpy as np
import optuna
import pandas as pd
import torch
import torch.utils.data
import torchvision
from optuna.trial import TrialState
from torch import Tensor
from tqdm import tqdm

from detectors.data import create_dataset
from detectors.eval import get_ood_results
from detectors.methods import Detector
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


class OODBenchmarkPipeline(Pipeline, ABC):
    def __init__(
        self,
        in_dataset_name: str,
        out_datasets_names_splits: Dict[str, Any],
        transform: Callable,
        batch_size: int,
        limit_fit: Optional[float] = 1.0,
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
    def _setup_datasets(self):
        """Setup `in_dataset`, `out_dataset`, `fit_dataset` and `out_datasets`."""
        ...

    def _setup_dataloaders(self):
        if self.fit_dataset is None or self.in_dataset is None or self.out_datasets is None:
            raise ValueError("Datasets are not set.")

        if self.limit_fit is None:
            self.limit_fit = 1.0
        self.limit_fit = min(int(self.limit_fit * len(self.fit_dataset)), len(self.fit_dataset))

        # random indices
        subset = np.random.choice(np.arange(len(self.fit_dataset)), self.limit_fit, replace=False).tolist()
        self.fit_dataset = torch.utils.data.Subset(self.fit_dataset, subset)
        self.fit_dataloader = torch.utils.data.DataLoader(
            self.fit_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

        test_dataset = torch.utils.data.ConcatDataset([self.in_dataset, self.out_dataset])
        test_labels = torch.utils.data.TensorDataset(
            torch.cat(
                [torch.zeros(len(self.in_dataset))]  # type: ignore
                + [torch.ones(len(d)) * (i + 1) for i, d in enumerate(self.out_datasets.values())]  # type: ignore
            ).long()
        )

        test_dataset = ConcatDatasetsDim1([test_dataset, test_labels])
        # shuffle and subsample test_dataset
        # test_dataset = torch.utils.data.Subset(test_dataset, np.random.permutation(len(test_dataset)).tolist())
        # test_dataset = torch.utils.data.Subset(test_dataset, np.arange(10_000).tolist())
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

        self.fit_dataloader = self.accelerator.prepare(self.fit_dataloader)
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

        _logger.info(f"Using {len(self.fit_dataset)} samples for fitting.")
        _logger.info(f"Using {len(test_dataset)} samples for testing.")

    def setup(self):
        self._setup_datasets()
        self._setup_dataloaders()

    def preprocess(self, method: Detector) -> Detector:
        if method.model is not None:
            _logger.info("Preparing model...")
            method.model.eval()
            method.model = self.accelerator.prepare(method.model)

        if self.fit_dataset is None:
            _logger.warning("Fit dataset is not set or not supported. Returning.")
            return method

        if not hasattr(method.detector, "update"):
            _logger.warning("Detector does not support fitting. Returning.")
            return method

        progress_bar = tqdm(
            range(len(self.fit_dataloader)), desc="Fitting", disable=not self.accelerator.is_local_main_process
        )
        fit_length = len(self.fit_dataloader.dataset)
        example = next(iter(self.fit_dataloader))[0]
        method.start(example=example, fit_length=fit_length)
        for x, y in self.fit_dataloader:
            method.update(x, y)
            progress_bar.update(1)
        progress_bar.close()
        self.accelerator.wait_for_everyone()
        method.end()
        return method

    def run(self, method: Detector) -> Dict[str, Any]:
        self.method = method

        _logger.info("Running pipeline...")
        self.preprocess(self.method)

        # initialize based on dataset size
        self.infer_times = torch.empty(len(self.test_dataloader.dataset), dtype=torch.float32)
        test_labels = torch.empty(len(self.test_dataloader.dataset), dtype=torch.int64)
        test_scores = torch.empty(len(self.test_dataloader.dataset), dtype=torch.float32)
        self.infer_times = []
        # test_labels = []
        # test_scores = []
        idx = 0
        progress_bar = tqdm(
            range(len(self.test_dataloader)), desc="Inference", disable=not self.accelerator.is_local_main_process
        )
        for x, y, labels in self.test_dataloader:
            t1 = time.time()
            score = self.method(x)
            t2 = time.time()

            labels, score = self.accelerator.gather_for_metrics((labels, score))

            # test_labels.append(labels.cpu())
            # test_scores.append(score.detach().cpu())
            self.infer_times.append(t2 - t1)
            test_labels[idx : idx + x.shape[0]] = labels.cpu()
            test_scores[idx : idx + x.shape[0]] = score.detach().cpu()

            idx += x.shape[0]
            progress_bar.update(1)
        progress_bar.close()

        self.accelerator.wait_for_everyone()
        self.infer_times = np.mean(self.infer_times)
        # test_scores = torch.cat(test_scores).view(-1)
        # test_labels = torch.cat(test_labels).view(-1)

        res_obj = self.postprocess(test_scores, test_labels)

        return {"results": res_obj, "scores": test_scores, "labels": test_labels}

    def postprocess(self, test_scores: Tensor, test_labels: Tensor):
        _logger.info("Computing metrics...")
        in_scores = test_scores[test_labels == 0]

        results = {}
        for i, ood_dataset_name in enumerate(self.out_datasets_names):
            ood_scores = test_scores[test_labels == (i + 1)]
            results[ood_dataset_name] = get_ood_results(in_scores, ood_scores)
            results[ood_dataset_name]["time"] = self.infer_times

        results["average"] = {
            k: np.mean([results[ds][k] for ds in self.out_datasets_names])
            for k in results[self.out_datasets_names[0]].keys()
        }
        results["average"]["time"] = self.infer_times
        ood_scores = test_scores[test_labels > 0]

        return results

    def report(self, results: Dict[str, Dict[str, Any]]) -> str:
        # log results in a table
        df = pd.DataFrame()

        for ood_dataset, res in results.items():
            df = pd.concat([df, pd.DataFrame(res, index=[ood_dataset])])
        df.columns = [METRICS_NAMES_PRETTY[k] for k in df.columns]
        return df.to_string(index=True, float_format="{:.4f}".format)


@register_pipeline("ood_cifar10_benchmark")
class OODCifar10BenchmarkPipeline(OODBenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=1, batch_size=128, seed=42, **kwargs) -> None:
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
                "uniform": None,
                "gaussian": None,
            },
            transform=transform,
            batch_size=batch_size,
            limit_fit=limit_fit,
            seed=seed,
        )

    def _setup_datasets(self):
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
class OODCifar100BenchmarkPipeline(OODBenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=1, batch_size=128, seed=42, **kwargs) -> None:
        super().__init__(
            "cifar100",
            {
                "cifar10": "test",
                "svhn": "test",
                "isun": None,
                "lsun_c": None,
                "lsun_r": None,
                "tiny_imagenet_c": None,
                "tiny_imagenet_r": None,
                "textures": None,
                "places365": None,
                "english_chars": None,
                "uniform": None,
                "gaussian": None,
            },
            transform=transform,
            batch_size=batch_size,
            limit_fit=limit_fit,
            seed=seed,
        )

    def _setup_datasets(self):
        _logger.info("Loading In-distribution dataset...")
        self.fit_dataset = create_dataset(self.in_dataset_name, split="train", transform=self.transform, download=True)
        self.in_dataset = create_dataset(self.in_dataset_name, split="test", transform=self.transform, download=True)

        _logger.info("Loading OOD datasets...")
        self.out_datasets = {
            ds: create_dataset(ds, split=split, transform=self.transform, download=True)
            for ds, split in self.out_datasets_names_splits.items()
        }
        self.out_dataset = torch.utils.data.ConcatDataset(list(self.out_datasets.values()))


@register_pipeline("ood_imagenet_benchmark")
class OODImageNetBenchmarkPipeline(OODBenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=1, batch_size=64, seed=42, **kwargs) -> None:
        super().__init__(
            "ilsvrc2012",
            {
                "mos_inaturalist": None,
                "mos_sun": None,
                "mos_places365": None,
                "textures": None,
                "imagenet_o": None,
                "openimage_o": None,
                "imagenet_a": None,
                "imagenet_r": None,
                "uniform": None,
                "gaussian": None,
            },
            limit_fit=limit_fit,
            transform=transform,
            batch_size=batch_size,
            seed=seed,
        )

    def _setup_datasets(self):
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
    def __init__(self, transform: Callable, limit_fit=1, batch_size=64) -> None:
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

    def _setup_datasets(self):
        _logger.info("Loading In-distribution dataset...")
        self.transform.transforms.append(torchvision.transforms.Grayscale(num_output_channels=1))
        self.fit_dataset = create_dataset(self.in_dataset_name, split="train", transform=self.transform)
        self.in_dataset = create_dataset(self.in_dataset_name, split="test", transform=self.transform)

        _logger.info("Loading OOD datasets...")
        self.out_datasets = {
            ds: create_dataset(ds, split=split, transform=self.transform, download=True)
            for ds, split in self.out_datasets_names_splits.items()
        }
        self.out_dataset = torch.utils.data.ConcatDataset(list(self.out_datasets.values()))


class OODValidationPipeline(OODBenchmarkPipeline, ABC):
    def run(
        self,
        method: Detector,
        hyperparameters: Dict[str, List[Any]],
        objective_metric: Literal["fpr_at_0.95_tpr", "auroc"] = "fpr_at_0.95_tpr",
        n_trials=20,
    ) -> Dict[str, Any]:
        self.method = method
        self.hyperparameters = hyperparameters
        self.objective_metric = objective_metric

        direction = "maximize" if objective_metric == "auroc" else "minimize"
        study = optuna.create_study(direction=direction)
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        self.method = self.method.set_params(**study.best_params)
        return {"method": self.method, "study": study}

    def objective(self, trial: optuna.trial.Trial) -> float:
        # build detector from trial params
        new_params = {}
        for k in self.hyperparameters:
            new_params[k] = trial.suggest_categorical(k, self.hyperparameters[k])
        self.method = self.method.set_params(**new_params)

        run_obj = super().run(self.method)
        results = run_obj["results"]
        return results["average"][self.objective_metric]

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


@register_pipeline("ood_cifar10_noise_validation")
class OODCifar10ValidationPipeline(OODValidationPipeline):
    def __init__(self, transform: Callable, limit_fit=1, batch_size=128, seed=42, **kwargs) -> None:
        super().__init__(
            "cifar10",
            {
                "uniform": None,
                "gaussian": None,
            },
            transform=transform,
            batch_size=batch_size,
            limit_fit=limit_fit,
            seed=seed,
        )

    def _setup_datasets(self):
        _logger.info("Loading In-distribution dataset...")
        self.fit_dataset = create_dataset(self.in_dataset_name, split="train", transform=self.transform)
        self.in_dataset = create_dataset(self.in_dataset_name, split="test", transform=self.transform)

        _logger.info("Loading OOD datasets...")
        self.out_datasets = {
            ds: create_dataset(ds, split=split, transform=self.transform, download=True)
            for ds, split in self.out_datasets_names_splits.items()
        }
        self.out_dataset = torch.utils.data.ConcatDataset(list(self.out_datasets.values()))


@register_pipeline("ood_cifar100_noise_validation")
class OODCifar100ValidationPipeline(OODValidationPipeline):
    def __init__(self, transform: Callable, limit_fit=1, batch_size=128, seed=42, **kwargs) -> None:
        super().__init__(
            "cifar100",
            {
                "uniform": None,
                "gaussian": None,
            },
            transform=transform,
            batch_size=batch_size,
            limit_fit=limit_fit,
            seed=seed,
        )

    def _setup_datasets(self):
        _logger.info("Loading In-distribution dataset...")
        self.fit_dataset = create_dataset(self.in_dataset_name, split="train", transform=self.transform)
        self.in_dataset = create_dataset(self.in_dataset_name, split="test", transform=self.transform)

        _logger.info("Loading OOD datasets...")
        self.out_datasets = {
            ds: create_dataset(ds, split=split, transform=self.transform, download=True)
            for ds, split in self.out_datasets_names_splits.items()
        }
        self.out_dataset = torch.utils.data.ConcatDataset(list(self.out_datasets.values()))
