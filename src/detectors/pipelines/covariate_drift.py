import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import accelerate
import torch
import torch.utils.data
from torch import Tensor
from tqdm import tqdm

from detectors.data import create_dataset
from detectors.pipelines import register_pipeline
from detectors.pipelines.base import Pipeline
from detectors.utils import ConcatDatasetsDim1

_logger = logging.getLogger(__name__)


class CovariateDriftBasePipeline(Pipeline):
    def __init__(
        self,
        in_dataset: torch.utils.data.Dataset,
        out_dataset: torch.utils.data.Dataset,
        fit_dataset: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 128,
    ) -> None:
        self.in_dataset = in_dataset
        self.out_dataset = out_dataset
        self.fit_dataset = fit_dataset
        self.batch_size = batch_size

        self.accelerator = accelerate.Accelerator()
        super().__init__()

    def setup(self):
        test_dataset = torch.utils.data.ConcatDataset([self.in_dataset, self.out_dataset])
        test_labels = torch.utils.data.TensorDataset(
            torch.cat([torch.zeros(len(self.in_dataset))] + [torch.ones(len(self.out_dataset))]).long()  # type: ignore
        )

        self.test_dataset = ConcatDatasetsDim1([test_dataset, test_labels])
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        if self.fit_dataset is not None:
            self.fit_dataloader = torch.utils.data.DataLoader(
                self.fit_dataset, batch_size=self.batch_size, shuffle=True
            )
            self.fit_dataloader = self.accelerator.prepare(self.fit_dataloader)
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

    def preprocess(self, method):
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

    def run(self, method, model):
        model.eval()
        self.method = method
        if hasattr(method.detector, "update"):
            self.preprocess(method)
        test_labels = []
        test_targets = []
        test_scores = []
        test_preds = []
        progress_bar = tqdm(
            range(len(self.test_dataloader)), desc="Inference", disable=not self.accelerator.is_local_main_process
        )
        for x, y, labels in self.test_dataloader:
            test_labels.append(labels.cpu())
            test_targets.append(y.cpu())

            s = self.method(x)
            with torch.no_grad():
                logits = model(x)
            test_scores.append(s.detach().cpu())
            test_preds.append(logits.detach().cpu().argmax(1))

            progress_bar.update(1)

        self.accelerator.wait_for_everyone()

        test_labels = torch.cat(test_labels).view(-1)
        test_targets = torch.cat(test_targets).view(-1)
        test_scores = torch.cat(test_scores).view(-1)
        test_preds = torch.cat(test_preds).view(-1)

        _logger.info("Computing metrics...")
        metrics = self.postprocess(test_scores, test_preds, test_targets, test_labels)

        return {
            "method": self.method,
            "test_scores": test_scores,
            "test_preds": test_preds,
            "test_targets": test_targets,
            "test_labels": test_labels,
            **metrics,
        }

    def postprocess(
        self, test_scores: Tensor, test_preds: Tensor, test_targets: Tensor, test_labels: Tensor
    ) -> Dict[str, Any]:
        return {}

    @classmethod
    def report(cls, results: Dict[str, Any]):
        # log results
        return


class CovariateDriftPipeline(CovariateDriftBasePipeline):
    def __init__(
        self,
        dataset_name: str,
        dataset_splits: List[str],
        transform,
        corruptions: List[str],
        intensities: List[int],
        batch_size: int = 128,
        warmup_size=1000,
        **kwargs
    ) -> None:
        """Covariate Drift Pipeline Factory

        Args:
            dataset_name (str): Name of the dataset
            dataset_splits (List[str]): List of dataset splits to use
            transform (Callable): Transform to apply to the dataset
            corruptions (List[str]): List of corruptions to apply
            intensities (List[int]): List of intensities to apply
            batch_size (int, optional): Batch size. Defaults to 128.

        Returns:
            CovariateDriftPipeline: Covariate Drift Pipeline

        Example:
        """
        _logger.info("Creating datasets...")

        fit_dataset = create_dataset(dataset_name, split=dataset_splits[0], transform=transform)
        in_dataset = create_dataset(dataset_name, split=dataset_splits[1], transform=transform)
        max_dataset_size = len(in_dataset) // (len(corruptions) * len(intensities) + 1)
        splits = torch.arange(0, len(in_dataset), max_dataset_size)
        _logger.info("Splits are: %s", splits)
        in_dataset = torch.utils.data.Subset(in_dataset, torch.arange(0, splits[1].item()).numpy())
        warmup_dataset = torch.utils.data.Subset(fit_dataset, torch.randperm(len(fit_dataset))[:warmup_size].numpy())
        in_dataset = torch.utils.data.ConcatDataset([warmup_dataset, in_dataset])
        out_datasets = {}
        for i, corruption in enumerate(corruptions):
            out_datasets[corruption] = []
            for j, intensity in enumerate(intensities):
                indices = torch.arange(splits[i + j + 1].item(), splits[i + j + 2].item(), dtype=torch.int).numpy()
                out_datasets[corruption].append(
                    torch.utils.data.Subset(
                        create_dataset(dataset_name + "c", split=corruption, intensity=intensity, transform=transform),
                        indices,
                    )
                )

        # increasing intensity concat dataset
        out_dataset = []
        for i, intensity in enumerate(intensities):
            # shuffle corruptions?
            out_dataset.append(
                torch.utils.data.ConcatDataset([out_datasets[corruption][i] for corruption in corruptions])
            )
        out_dataset = torch.utils.data.ConcatDataset(out_dataset)

        _logger.debug("Fit dataset size: %s", {len(fit_dataset)})
        _logger.debug("In dataset size: %s", {len(in_dataset)})
        _logger.debug("Out dataset size: %s", {len(out_dataset)})

        super().__init__(in_dataset, out_dataset, fit_dataset, batch_size)


@register_pipeline("one_corruption_covariate_drift_cifar10")
class OneCorruptionCovariateDriftCifar10Pipeline(CovariateDriftPipeline):
    def __init__(self, transform, corruption: str, intensities: List[int], batch_size: int = 128, **kwargs) -> None:
        super().__init__("cifar10", ["train", "test"], transform, [corruption], intensities, batch_size, **kwargs)


@register_pipeline("one_corruption_covariate_drift_cifar100")
class OneCorruptionCovariateDriftCifar100Pipeline(CovariateDriftPipeline):
    def __init__(self, transform, corruption: str, intensities: List[int], batch_size: int = 128, **kwargs) -> None:
        super().__init__("cifar100", ["train", "test"], transform, [corruption], intensities, batch_size, **kwargs)
