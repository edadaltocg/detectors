import logging
import time
from typing import Any, Callable, Dict, Union

import accelerate
import numpy as np
import torch
import torch.utils.data
from torch import Tensor
from tqdm import tqdm

from detectors.data import create_dataset
from detectors.eval import get_ood_results, risks_coverages_selective_net
from detectors.methods.templates import Detector, DetectorWrapper
from detectors.pipelines import register_pipeline
from detectors.pipelines.ood import OODBenchmarkPipeline
from detectors.pipelines.sc import SelectiveClassificationPipeline
from detectors.scod_eval import plugin_bb

_logger = logging.getLogger(__name__)


class SCODPipeline(SelectiveClassificationPipeline, OODBenchmarkPipeline):
    """Selective Classification and OOD Detection Benchmark pipeline.

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
    """

    def __init__(
        self,
        in_dataset_name: str,
        out_datasets_names_splits: Dict[str, Any],
        transform: Callable,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        limit_fit: float = 0.0,
        limit_run: float = 1.0,
        seed: int = 42,
        accelerator=None,
    ):
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

        if self.limit_fit is None:
            self.limit_fit = 0
        accelerate.utils.set_seed(seed)
        self.setup()

    def _setup_datasets(self):
        SelectiveClassificationPipeline._setup_datasets(self)

        _logger.info("Loading OOD datasets...")
        self.out_datasets = {
            ds: create_dataset(ds, split=split, transform=self.transform, download=True)
            for ds, split in self.out_datasets_names_splits.items()
        }
        self.out_dataset = torch.utils.data.ConcatDataset(list(self.out_datasets.values()))

    def _setup_dataloaders(self):
        OODBenchmarkPipeline._setup_dataloaders(self)

    def setup(self, *args, **kwargs):
        self._setup_datasets()
        self._setup_dataloaders()

    def preprocess(self, method: Union[DetectorWrapper, Detector]) -> Union[DetectorWrapper, Detector]:
        return OODBenchmarkPipeline.preprocess(self, method)

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
        msp_scores = torch.empty(dataset_size, dtype=torch.float32)
        test_ds_idxs = torch.empty(dataset_size, dtype=torch.int64)
        _logger.debug("test_labels shape: %s", test_labels.shape)
        _logger.debug("test_scores shape: %s", test_scores.shape)

        self.infer_times = []
        idx = 0
        disable = False
        if self.accelerator is not None:
            disable = not self.accelerator.is_main_process
        progress_bar = tqdm(range(len(self.test_dataloader)), desc="Inference", disable=disable)
        for x, y, z in self.test_dataloader:
            x = x.to(device)
            t1 = time.time()
            score = self.method(x)
            t2 = time.time()
            with torch.no_grad():
                logits = self.model(x)
                msp, pred = torch.softmax(logits, dim=1).max(1)
                msp, pred = msp.cpu(), pred.cpu()
            labels = torch.where(z == 0, pred != y, 1).int()
            dataset_idxs = z

            if self.accelerator is not None:
                score = self.accelerator.gather_for_metrics(score)
                labels = self.accelerator.gather_for_metrics(labels)
                msp = self.accelerator.gather_for_metrics(msp)
                dataset_idxs = self.accelerator.gather_for_metrics(dataset_idxs)

            self.infer_times.append(t2 - t1)
            test_labels[idx : idx + labels.shape[0]] = labels.cpu()
            test_scores[idx : idx + score.shape[0]] = score.cpu()
            msp_scores[idx : idx + msp.shape[0]] = msp.cpu()
            test_ds_idxs[idx : idx + dataset_idxs.shape[0]] = dataset_idxs.cpu()

            idx += labels.shape[0]
            progress_bar.update(1)
        progress_bar.close()
        self.infer_times = np.mean(self.infer_times)
        test_scores = test_scores[:idx]
        msp_scores = msp_scores[:idx]
        test_labels = test_labels[:idx]
        test_ds_idxs = test_ds_idxs[:idx]
        res_obj = self.postprocess(msp_scores, test_scores, test_labels, test_ds_idxs)

        return {
            "results": res_obj,
            "scores": test_scores,
            "msp_scores": msp_scores,
            "labels": test_labels,
            "idxs": test_ds_idxs,
        }

    def postprocess(
        self, msp_scores: Tensor, test_scores: Tensor, test_labels: Tensor, test_ds_idxs: Tensor
    ) -> Dict[str, Any]:
        _logger.info("Computing metrics...")

        in_scores = test_scores[test_labels == 0]
        ood_scores_in = test_scores[test_ds_idxs == 0][test_labels[test_ds_idxs == 0] == 1]
        msp_scores_in = msp_scores[test_labels == 0]
        msp_scores_ood = msp_scores[test_ds_idxs == 0][test_labels[test_ds_idxs == 0] == 1]

        results = {}
        for i, ood_dataset_name in enumerate(tqdm(self.out_datasets_names)):
            # print("\n---------------->", ood_dataset_name)
            ood_scores = test_scores[test_ds_idxs == (i + 1)]
            ood_scores = torch.cat([ood_scores, ood_scores_in], dim=0)
            msp_ood_scores = msp_scores[test_ds_idxs == (i + 1)]
            msp_ood_scores = torch.cat([msp_ood_scores, msp_scores_ood], dim=0)

            tscores = torch.cat([in_scores, ood_scores], dim=0)
            mscores = torch.cat([msp_scores_in, msp_ood_scores], dim=0)
            tlabels = torch.cat([torch.zeros_like(in_scores), torch.ones_like(ood_scores)], dim=0)
            risks, coverages, thrs = risks_coverages_selective_net(tscores, tlabels)
            auc = torch.trapz(coverages, risks).item()
            pi = sum(test_ds_idxs == 0) / (sum(test_ds_idxs == (i + 1)) + sum(test_ds_idxs == 0))
            pi = pi.item()
            pbb_risks, pbb_coverages, pbb_thrs = plugin_bb(mscores.numpy(), tscores.numpy(), tlabels.numpy(), pi=pi)
            auc_pbb = np.trapz(pbb_coverages, pbb_risks)
            # risks = risks.numpy().tolist()
            # coverages = coverages.numpy().tolist()
            # thrs = thrs.numpy().tolist()
            results[ood_dataset_name] = get_ood_results(in_scores, test_scores[test_ds_idxs == (i + 1)])
            results[ood_dataset_name]["aurc"] = auc
            results[ood_dataset_name]["aurc_pbb"] = auc_pbb
            results[ood_dataset_name]["time"] = self.infer_times

            # print(pbb_coverages, pbb_risks)

        # in-distribution dataset performance
        ood_scores = ood_scores_in
        tscores = torch.cat([in_scores, ood_scores], dim=0)
        mscores = torch.cat([msp_scores_in, msp_scores_ood], dim=0)
        tlabels = torch.cat([torch.zeros_like(in_scores), torch.ones_like(ood_scores)], dim=0)
        risks, coverages, thrs = risks_coverages_selective_net(tscores, tlabels)
        auc = torch.trapz(coverages, risks).item()
        # print("RC", coverages, risks)

        pbb_risks, pbb_coverages, pbb_thrs = plugin_bb(mscores.numpy(), tscores.numpy(), tlabels.numpy())
        auc_pbb = np.trapz(pbb_coverages, pbb_risks)
        # print("PBBRC", pbb_coverages, pbb_risks)

        # risks = risks.numpy().tolist()
        # coverages = coverages.numpy().tolist()
        # thrs = thrs.numpy().tolist()
        results[self.in_dataset_name] = get_ood_results(in_scores, ood_scores)
        results[self.in_dataset_name]["aurc"] = auc
        results[self.in_dataset_name]["aurc_pbb"] = auc_pbb
        results[self.in_dataset_name]["acc"] = len(in_scores) / (len(in_scores) + len(ood_scores))
        results[self.in_dataset_name]["time"] = self.infer_times

        results["average"] = {
            k: np.mean([results[ds][k] for ds in self.out_datasets_names])
            for k in results[self.out_datasets_names[0]].keys()
        }
        results["average"]["time"] = self.infer_times

        return results

    def report(self, results: Dict[str, Dict[str, Any]]) -> str:
        return OODBenchmarkPipeline.report(self, results)


@register_pipeline("scod_benchmark_cifar10")
class SCODCifar10BenchmarkPipeline(SCODPipeline):
    def __init__(self, transform: Callable, limit_fit=0.0, limit_run=1.0, batch_size=128, seed=42, **kwargs) -> None:
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
                "uniform": None,
                "gaussian": None,
            },
            transform=transform,
            batch_size=batch_size,
            limit_fit=limit_fit,
            limit_run=limit_run,
            seed=seed,
            **kwargs
        )


@register_pipeline("scod_benchmark_cifar100")
class SCODCifar100BenchmarkPipeline(SCODPipeline):
    def __init__(self, transform: Callable, limit_fit=0.0, limit_run=1.0, batch_size=128, seed=42, **kwargs) -> None:
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
                "uniform": None,
                "gaussian": None,
            },
            transform=transform,
            batch_size=batch_size,
            limit_fit=limit_fit,
            limit_run=limit_run,
            seed=seed,
            **kwargs
        )
