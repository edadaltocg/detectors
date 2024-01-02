import json
import logging
import time
from typing import Any, Callable, Dict, Union

import accelerate
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import Tensor
from tqdm import tqdm

from detectors.data import create_dataset
from detectors.eval import get_ood_results, risks_coverages_selective_net
from detectors.methods.templates import Detector, DetectorWrapper
from detectors.pipelines import register_pipeline
from detectors.pipelines.base import Pipeline

_logger = logging.getLogger(__name__)


class SelectiveClassificationPipeline(Pipeline):
    """Selective Classification Benchmark pipeline for post-hoc methods.

    Args:
        in_dataset_name (str): Name of the in-distribution dataset.
        transform (Callable): Transform to apply to the datasets.
        batch_size (int): Batch size.
        num_workers (int, optional): Number of workers. Defaults to 4.
        pin_memory (bool, optional): Pin memory. Defaults to True.
        prefetch_factor (int, optional): Prefetch factor. Defaults to 2.
        limit_fit (float, optional): Fraction of the training set to use for fitting. Defaults to 1.0.
        limit_run (float, optional): Fraction of the testing set to use for running. Defaults to 1.0.
        seed (int, optional): Random seed. Defaults to 42.
    """

    def __init__(
        self,
        in_dataset_name: str,
        transform: Any,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        limit_fit: float = 0,
        limit_run: float = 1.0,
        seed: int = 42,
        accelerator=None,
        **kwargs
    ) -> None:
        self.in_dataset_name = in_dataset_name
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.limit_fit = limit_fit
        self.limit_run = limit_run
        self.seed = seed
        self.accelerator = accelerator

        if self.limit_fit is None:
            self.limit_fit = 0
        accelerate.utils.set_seed(seed)
        self.setup()

    def _setup_datasets(self):
        _logger.info("Loading In-distribution dataset...")
        if self.limit_fit > 0:
            self.fit_dataset = create_dataset(
                self.in_dataset_name, split="train", transform=self.transform, download=True
            )
        else:
            self.fit_dataset = None
        self.in_dataset = create_dataset(self.in_dataset_name, split="test", transform=self.transform, download=True)

    def _setup_dataloaders(self):
        if self.in_dataset is None:
            raise ValueError("Datasets are not set.")

        if self.fit_dataset is not None:
            self.limit_fit = min(int(self.limit_fit * len(self.fit_dataset)), len(self.fit_dataset))

        # random indices
        if self.limit_run > 0 and self.fit_dataset is not None:
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

        self.test_dataloader = torch.utils.data.DataLoader(
            self.in_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def setup(self, *args, **kwargs):
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

    def postprocess(self, test_scores: Tensor, test_labels: Tensor):
        _logger.info("Computing metrics...")
        print(test_scores, test_labels)
        risks, coverages, thrs = risks_coverages_selective_net(test_scores, test_labels)
        auc = torch.trapz(coverages, risks).item()
        print(coverages, risks)
        risks = risks.numpy().tolist()
        coverages = coverages.numpy().tolist()
        thrs = thrs.numpy().tolist()
        in_scores = test_scores[test_labels == 0]  # correctly classified samples
        ood_score = test_scores[test_labels == 1]  # misclassified samples
        ood_res = get_ood_results(in_scores, ood_score)
        results = {
            "aurc": auc,
            **ood_res,
            "risk": json.dumps(risks),
            "coverage": json.dumps(coverages),
            "thr": json.dumps(thrs),
            "time": self.infer_times,
        }

        return results

    def run(self, method: Union[DetectorWrapper, Detector], model: torch.nn.Module) -> Dict[str, Any]:
        self.method = method
        self.model = model
        device = next(model.parameters()).device

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
        for x, y in self.test_dataloader:
            x = x.to(device)
            t1 = time.time()
            score = self.method(x)
            t2 = time.time()
            with torch.no_grad():
                pred = self.model(x).argmax(1).cpu()
            labels = (pred != y).int()

            if self.accelerator is not None:
                score = self.accelerator.gather_for_metrics(score)
                labels = self.accelerator.gather_for_metrics(labels)

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

    def report(self, results: Dict[str, Dict[str, Any]]) -> str:
        # log results in a table
        if "results" in results:
            results = results["results"]
        d = {k: [v] for k, v in results.items() if isinstance(v, float)}
        df = pd.DataFrame.from_dict(d, orient="columns")
        return df.to_string(index=True, float_format="{:.4f}".format)


@register_pipeline("sc_benchmark_cifar10")
class SCCifar10BenchmarkPipeline(SelectiveClassificationPipeline):
    def __init__(self, transform: Callable, limit_fit=1.0, limit_run=1.0, batch_size=128, seed=42, **kwargs) -> None:
        super().__init__(
            "cifar10", transform=transform, batch_size=batch_size, limit_fit=limit_fit, limit_run=limit_run, seed=seed
        )


@register_pipeline("sc_benchmark_cifar100")
class SCCifar100BenchmarkPipeline(SelectiveClassificationPipeline):
    def __init__(self, transform: Callable, limit_fit=1.0, limit_run=1.0, batch_size=128, seed=42, **kwargs) -> None:
        super().__init__(
            "cifar100", transform=transform, batch_size=batch_size, limit_fit=limit_fit, limit_run=limit_run, seed=seed
        )


@register_pipeline("sc_benchmark_imagenet")
class SCImageNetBenchmarkPipeline(SelectiveClassificationPipeline):
    def __init__(self, transform: Callable, limit_fit=1.0, limit_run=1.0, batch_size=128, seed=42, **kwargs) -> None:
        super().__init__(
            "imagenet", transform=transform, batch_size=batch_size, limit_fit=limit_fit, limit_run=limit_run, seed=seed
        )
