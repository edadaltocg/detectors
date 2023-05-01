import logging
from typing import Callable, Optional

import accelerate
import numpy as np
import torch
import torch.utils.data

import detectors
from detectors.pipelines import register_pipeline
from detectors.pipelines.base import Pipeline
from detectors.pipelines.ood import OODBenchmarkPipeline
from detectors.utils import ConcatDatasetsDim1

_logger = logging.getLogger(__name__)


@register_pipeline("osr_cifar10")
class OSRCifar10(OODBenchmarkPipeline):
    def __init__(
        self, transform: Callable, batch_size: int, limit_fit: Optional[int] = None, seed: int = 42, **kwargs
    ) -> None:
        super().__init__(
            "cifar10", {"cifar100": "test"}, transform, batch_size, limit_fit=limit_fit, seed=seed, **kwargs
        )


@register_pipeline("osr_cifar100")
class OSRCifar100(OODBenchmarkPipeline):
    def __init__(
        self, transform: Callable, batch_size: int, limit_fit: Optional[int] = None, seed: int = 42, **kwargs
    ) -> None:
        super().__init__(
            "cifar100", {"cifar10": "test"}, transform, batch_size, limit_fit=limit_fit, seed=seed, **kwargs
        )


@register_pipeline("osr_imagenet")
class OSRImagenet(OODBenchmarkPipeline):
    def __init__(
        self, transform: Callable, batch_size: int, limit_fit: Optional[int] = None, seed: int = 42, **kwargs
    ) -> None:
        super().__init__(
            "imagenet",
            {
                "imagenet_o": None,
            },
            transform,
            batch_size,
            limit_fit=limit_fit,
            seed=seed,
            **kwargs,
        )


@register_pipeline("one_class_versus_others_cifar10")
class SingleClassCifar10(Pipeline):
    # TODO
    def __init__(
        self,
        in_dataset_name: str,
        in_dataset_split: str,
        transform: Callable,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        limit_fit: float = 1.0,
        limit_run: float = 1.0,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.transform = transform
        self.batch_size = batch_size
        self.limit_fit = limit_fit
        self.limit_run = limit_run
        self.seed = seed
        self.kwargs = kwargs
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.in_dataset_name = in_dataset_name
        self.in_dataset_split = in_dataset_split

        self.in_dataset_name = "cifar10"
        accelerate.utils.set_seed(seed)
        self.setup()

    def _setup_datasets(self):
        """Setup `in_dataset`."""
        self.in_dataset = detectors.create_dataset(
            self.in_dataset_name, transform=self.transform, split=self.in_dataset_split
        )

    def _setup_dataloaders(self):
        if self.limit_fit is None:
            self.limit_fit = 1.0
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

        self.test_dataset = torch.utils.data.ConcatDataset([self.in_dataset])
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

        _logger.info(f"Using {len(self.fit_dataset)} samples for fitting.")
        _logger.info(f"Using {len(self.test_dataset)} samples for testing.")

    def setup(self):
        self._setup_datasets()
        self._setup_dataloaders()
