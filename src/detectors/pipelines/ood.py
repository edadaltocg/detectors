"""
OOD Pipelines.
"""
import logging
import time
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

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
from detectors.eval import METRICS_NAMES_PRETTY, get_ood_results
from detectors.methods import DetectorWrapper
from detectors.methods.templates import Detector
from detectors.pipelines import register_pipeline
from detectors.pipelines.base import Pipeline
from detectors.utils import ConcatDatasetsDim1, sync_tensor_across_gpus

_logger = logging.getLogger(__name__)


class OODBenchmarkPipeline(Pipeline):
    """OOD Benchmark pipeline.

    Args:
        in_dataset_name (str): Name of the in-distribution dataset.
        out_datasets_names_splits (Dict[str, Any]): Dictionary mapping out-distribution dataset names to their splits.
        transform (Callable): Transform to apply to the datasets.
        batch_size (int): Batch size.
        num_workers (int, optional): Number of workers. Defaults to 4.
        pin_memory (bool, optional): Pin memory. Defaults to True.
        prefetch_factor (int, optional): Prefetch factor. Defaults to 2.
        limit_fit (float, optional): Fraction of the training set to use for fitting. Defaults to 1.0.
        limit_run (float, optional): Fraction of the testing set to use for running. Defaults to 1.0.
        seed (int, optional): Random seed. Defaults to 42.
        accelerator (Any, optional): Accelerator. Defaults to None.
    """

    def __init__(
        self,
        in_dataset_name: str,
        out_datasets_names_splits: Dict[str, Any],
        transform: Callable,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        limit_fit: float = 1.0,
        limit_run: float = 1.0,
        seed: int = 42,
        accelerator=None,
    ) -> None:
        self.in_dataset_name = in_dataset_name
        self.out_datasets_names_splits = out_datasets_names_splits
        self.out_datasets_names = list(out_datasets_names_splits.keys())
        self.limit_fit = limit_fit
        self.limit_run = limit_run
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        self.accelerator = accelerator

        self.fit_dataset = None
        self.in_dataset = None
        self.out_dataset = None
        self.out_datasets = None

        accelerate.utils.set_seed(seed)
        print("Setting up datasets...")
        self.setup()

    def _setup_datasets(self):
        """Setup `in_dataset`, `out_dataset`, `fit_dataset` and `out_datasets`."""
        raise NotImplementedError

    def _setup_dataloaders(self):
        if self.in_dataset is None or self.out_datasets is None or self.out_dataset is None:
            raise ValueError("Datasets are not set.")

        if self.limit_fit is None or self.limit_fit <= 0:
            self.fit_dataset = None
            self.fit_dataloader = None
        else:
            self.limit_fit = min(int(self.limit_fit * len(self.fit_dataset)), len(self.fit_dataset))

            # random indices
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

        self.test_dataset = torch.utils.data.ConcatDataset([self.in_dataset, self.out_dataset])
        test_labels = torch.utils.data.TensorDataset(
            torch.cat(
                [torch.zeros(len(self.in_dataset))]  # type: ignore
                + [torch.ones(len(d)) * (i + 1) for i, d in enumerate(self.out_datasets.values())]  # type: ignore
            ).long()
        )

        self.test_dataset = ConcatDatasetsDim1([self.test_dataset, test_labels])
        # shuffle and subsample test_dataset
        subset = np.random.choice(
            np.arange(len(self.test_dataset)), int(self.limit_run * len(self.test_dataset)), replace=False
        ).tolist()
        self.test_dataset = torch.utils.data.Subset(self.test_dataset, subset)
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

        if self.accelerator is not None:
            self.fit_dataloader = self.accelerator.prepare(self.fit_dataloader)
            self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

        if self.fit_dataset is not None:
            _logger.info(f"Using {len(self.fit_dataset)} samples for fitting.")
        _logger.info(f"Using {len(self.test_dataset)} samples for testing.")

    def setup(self):
        self._setup_datasets()
        self._setup_dataloaders()

    def preprocess(self, method: Union[DetectorWrapper, Detector]) -> Union[DetectorWrapper, Detector]:
        if self.fit_dataset is None:
            _logger.warning("Fit dataset is not set or not supported. Returning.")
            return method

        if not hasattr(method.detector, "update"):
            _logger.warning("Detector does not support fitting. Returning.")
            return method

        disable = False
        if self.accelerator is not None:
            disable = not self.accelerator.is_main_process
        progress_bar = tqdm(range(len(self.fit_dataloader)), desc="Fitting", disable=disable)

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

        # initialize based on dataset size
        dataset_size = len(self.test_dataloader.dataset)
        test_labels = torch.empty(dataset_size, dtype=torch.int64)
        test_scores = torch.empty(dataset_size, dtype=torch.float32)
        _logger.debug("test_labels shape: %s", test_labels.shape)
        _logger.debug("test_scores shape: %s", test_scores.shape)

        self.infer_times = []
        idx = 0
        disable = False
        if self.accelerator is not None:
            disable = not self.accelerator.is_main_process
        progress_bar = tqdm(range(len(self.test_dataloader)), desc="Inference", disable=disable)
        for x, y, labels in self.test_dataloader:
            t1 = time.time()
            score = self.method(x)
            t2 = time.time()

            if self.accelerator is not None:
                score = self.accelerator.gather_for_metrics(score)
                labels = self.accelerator.gather_for_metrics(labels)
            # score = sync_tensor_across_gpus(score.detach())
            # labels = sync_tensor_across_gpus(labels.to(score.device))

            self.infer_times.append(t2 - t1)
            test_labels[idx : idx + labels.shape[0]] = labels.cpu()
            test_scores[idx : idx + score.shape[0]] = score.cpu()

            idx += labels.shape[0]
            progress_bar.update(1)
        progress_bar.close()
        self.infer_times = np.mean(self.infer_times)
        test_scores = test_scores[:idx]
        test_labels = test_labels[:idx]
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

        return results

    def report(self, results: Dict[str, Dict[str, Any]]) -> str:
        # log results in a table
        if "results" in results:
            results = results["results"]
        df = pd.DataFrame()

        for ood_dataset, res in results.items():
            df = pd.concat([df, pd.DataFrame(res, index=[ood_dataset])])
        df.columns = [METRICS_NAMES_PRETTY[k] for k in df.columns]
        return df.to_string(index=True, float_format="{:.4f}".format)


@register_pipeline("ood_benchmark_cifar10")
class OODCifar10BenchmarkPipeline(OODBenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=1.0, limit_run=1.0, batch_size=128, seed=42, **kwargs) -> None:
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
            limit_run=limit_run,
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


# @register_pipeline("ood_benchmark_cifar10_scood")  # TODO
# @register_pipeline("ood_benchmark_cifar10_mood")  # TODO


@register_pipeline("ood_benchmark_cifar100")
class OODCifar100BenchmarkPipeline(OODBenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=1.0, limit_run=1.0, batch_size=128, seed=42, **kwargs) -> None:
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
            limit_run=limit_run,
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


@register_pipeline("ood_benchmark_imagenet")
class OODImageNetBenchmarkPipeline(OODBenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=1.0, limit_run=1.0, batch_size=64, seed=42, **kwargs) -> None:
        super().__init__(
            "ilsvrc2012",
            {
                # "mos_inaturalist": None,
                # "mos_sun": None,
                # "mos_places365": None,
                # "textures": None,
                # "openimage_o": None,
                "imagenet_o": None,
                "ninco": None,
                "ssb_hard": None,
                "ssb_easy": None,
                "textures_clean": None,
                "places_clean": None,
                "inaturalist_clean": None,
                "openimage_o_clean": None,
                "species_clean": None,
            },
            limit_fit=limit_fit,
            limit_run=limit_run,
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


@register_pipeline("ood_benchmark_imagenet_reduced")
class OODImageNetBenchmarkPipelineReduced(OODBenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=1.0, limit_run=1.0, batch_size=64, seed=42, **kwargs) -> None:
        super().__init__(
            "ilsvrc2012",
            {
                "mos_inaturalist": None,
                "mos_sun": None,
                "mos_places365": None,
                "textures": None,
            },
            limit_fit=limit_fit,
            limit_run=limit_run,
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


@register_pipeline("ood_benchmark_imagenet_near")
class OODImageNetBenchmarkPipelineHard(OODBenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=1.0, limit_run=1.0, batch_size=64, seed=42, **kwargs) -> None:
        super().__init__(
            "ilsvrc2012",
            {
                "imagenet_r": None,
                "ninco": None,
                "ssb_hard": None,
            },
            limit_fit=limit_fit,
            limit_run=limit_run,
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


@register_pipeline("ood_benchmark_imagenet_all_2")
class OODImageNetBenchmarkPipelineAll(OODBenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=1.0, limit_run=1.0, batch_size=64, seed=42, **kwargs) -> None:
        super().__init__(
            "ilsvrc2012",
            {
                "inaturalist_clean": None,
                "species_clean": None,
                "places_clean": None,
                "openimage_o_clean": None,
                "ssb_easy": None,
                "textures_clean": None,
                "ninco": None,
                "ssb_hard": None,
            },
            limit_fit=limit_fit,
            limit_run=limit_run,
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


@register_pipeline("ood_benchmark_imagenet_far")
class OODImageNetBenchmarkPipelineEasy(OODBenchmarkPipeline):
    def __init__(self, transform: Callable, limit_fit=1.0, limit_run=1.0, batch_size=64, seed=42, **kwargs) -> None:
        super().__init__(
            "ilsvrc2012",
            {
                "inaturalist_clean": None,
                "species_clean": None,
                "places_clean": None,
                "openimage_o_clean": None,
                "ssb_easy": None,
            },
            limit_fit=limit_fit,
            limit_run=limit_run,
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


class OODValidationPipeline(OODBenchmarkPipeline):
    """Pipeline for OOD validation.

    This pipeline is used to validate the performance of a model on OOD datasets.

    Args:
        method (DetectorWrapper): The OOD detection method to use.
        hyperparameters (Dict[str, Union[List[Any], Tuple[Any], Dict[str, Any]]]): The hyperparameters to use for the method.
        objective_metric (Literal["fpr_at_0.95_tpr", "auroc"], optional): The metric to optimize. Defaults to "auroc".
        n_trials (int, optional): The number of trials to run. Defaults to 20.
    """

    # TODO: include prevent refit flag.

    def run(
        self,
        method: DetectorWrapper,
        hyperparameters: Dict[str, Union[List[Any], Tuple[Any], Dict[str, Any]]],
        objective_metric: Literal["fpr_at_0.95_tpr", "auroc"] = "auroc",
        objective_dataset: str = "average",
        n_trials=20,
    ) -> Dict[str, Any]:
        self.method = method
        self.hyperparameters = hyperparameters
        self.objective_metric = objective_metric
        self.objective_dataset = objective_dataset

        direction = "maximize" if objective_metric == "auroc" else "minimize"
        sampler = None
        if all(isinstance(v, (list, tuple)) for v in hyperparameters.values()):
            sampler = optuna.samplers.GridSampler(search_space=hyperparameters)
            lengths = np.array([len(v) for v in hyperparameters.values()])
            n_trials = min(int(np.prod(lengths)), n_trials)
        study = optuna.create_study(study_name="ood-val", sampler=sampler, direction=direction)
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        self.method.set_hyperparameters(**study.best_params)
        return {
            "method": self.method,
            "study": study,
            "best_params": study.best_params,
            "best_value": study.best_trial.value,
        }

    def objective(self, trial: optuna.trial.Trial) -> float:
        # build detector from trial params
        new_params = {}
        for k in self.hyperparameters:
            if isinstance(self.hyperparameters[k], (list, tuple)):
                new_params[k] = trial.suggest_categorical(k, self.hyperparameters[k])
            elif isinstance(self.hyperparameters[k], dict):
                step = self.hyperparameters[k]["step"]
                low = self.hyperparameters[k]["low"]
                high = self.hyperparameters[k]["high"]
                param_type = type(step)
                if param_type == float:
                    new_params[k] = trial.suggest_float(k, low=low, high=high, step=step)
                elif param_type == int:
                    new_params[k] = trial.suggest_int(k, low=low, high=high, step=step)

        self.method.set_hyperparameters(**new_params)
        # print methods params
        run_obj = super().run(self.method)
        results = run_obj["results"]
        return results[self.objective_dataset][self.objective_metric]

    def report(self, results: Dict[str, Any]):
        if "study" not in results:
            raise ValueError("The results dict must contain a 'study' key.")
        study = results["study"]
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


@register_pipeline("ood_validation_cifar10")
class OODCifar10ValidationPipeline(OODCifar10BenchmarkPipeline, OODValidationPipeline):
    def __init__(self, transform: Callable, limit_fit=1, limit_run=0.1, batch_size=128, seed=42, **kwargs) -> None:
        super().__init__(
            transform=transform, batch_size=batch_size, limit_fit=limit_fit, limit_run=limit_run, seed=seed
        )


@register_pipeline("ood_validation_noise_cifar10")
class OODCifar10NoiseValidationPipeline(OODValidationPipeline):
    def __init__(self, transform: Callable, limit_fit=1, limit_run=0.1, batch_size=128, seed=42, **kwargs) -> None:
        super().__init__(
            "cifar10",
            {
                "uniform": None,
                "gaussian": None,
            },
            transform=transform,
            batch_size=batch_size,
            limit_fit=limit_fit,
            limit_run=limit_run,
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


@register_pipeline("ood_validation_cifar100")
class OODCifar100ValidationPipeline(OODCifar100BenchmarkPipeline, OODValidationPipeline):
    def __init__(self, transform: Callable, limit_fit=1, limit_run=0.1, batch_size=128, seed=42, **kwargs) -> None:
        super().__init__(
            transform=transform, batch_size=batch_size, limit_fit=limit_fit, limit_run=limit_run, seed=seed
        )


@register_pipeline("ood_validation_noise_cifar100")
class OODCifar100NoiseValidationPipeline(OODValidationPipeline):
    def __init__(self, transform: Callable, limit_fit=1, limit_run=0.1, batch_size=128, seed=42, **kwargs) -> None:
        super().__init__(
            "cifar100",
            {
                "uniform": None,
                "gaussian": None,
            },
            transform=transform,
            batch_size=batch_size,
            limit_fit=limit_fit,
            limit_run=limit_run,
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


@register_pipeline("ood_validation_imagenet")
class OODImageNetValidationPipeline(OODImageNetBenchmarkPipeline, OODValidationPipeline):
    def __init__(self, transform: Callable, limit_fit=1, limit_run=0.1, batch_size=64, seed=42, **kwargs) -> None:
        super().__init__(
            transform=transform, batch_size=batch_size, limit_fit=limit_fit, limit_run=limit_run, seed=seed
        )
