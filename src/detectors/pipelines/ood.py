import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List

import torch
import torch.utils.data
import torchmetrics.functional as metrics
from detectors.pipelines.base import Pipeline
from detectors.utils import ConcatDatasetsDim1
from torch import Tensor

from ..data import get_dataset


logger = logging.getLogger(__name__)


def fpr_at_fixed_tpr(fprs, tprs, thresholds, tpr_level: float = 0.95):
    if all(tprs < tpr_level):
        raise ValueError(f"No threshold allows for TPR at least {tpr_level}.")
    idxs = [i for i, x in enumerate(tprs) if x >= tpr_level]
    idx = min(idxs)
    return fprs[idx], tprs[idx], thresholds[idx]


def tnr_at_fixed_tpr(fprs, tprs, thresholds, tpr_level: float = 0.95):
    return 1 - fpr_at_fixed_tpr(fprs, tprs, thresholds, tpr_level)[0]


def compute_detection_error(op_fpr, op_tpr, pos_ratio):
    """Return the misclassification probability when TPR is fixed."""
    # Get ratios of positives to negatives
    neg_ratio = 1 - pos_ratio
    # Get indexes of all TPR >= fixed tpr level
    detection_error = pos_ratio * (1 - op_tpr) + neg_ratio * op_fpr
    # Return the minimum detection error such that TPR >= 0.95
    return detection_error


METRICS_NAMES_PRETTY = {
    "fpr_at_0.95_tpr": "FPR at 95% TPR",
    "tnr_at_0.95_tpr": "TNR at 95% TPR",
    "detection_error": "Detection error",
    "auroc": "AUROC",
    "aupr_in": "AUPR in",
    "aupr_out": "AUPR out",
    "thr": "Threshold",
}


class OODPipeline(Pipeline):
    def __init__(
        self, in_dataset_name: str, ood_datasets_names: List[str], device, batch_size: int, num_workers: int
    ) -> None:
        self.in_dataset = in_dataset_name
        self.ood_datasets = ood_datasets_names
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _setup(self):
        logger.info("Loading In-distribution dataset...")
        in_distribution_train = get_dataset(self.in_dataset, split="train", download=True)
        in_distribution_test = get_dataset(self.in_dataset, split="test", download=True)

        logger.info("Loading OOD datasets...")
        out_distribution_datasets = {ds: get_dataset(ds, split="test", download=True) for ds in self.ood_datasets}

        self.fit_dataloader = torch.utils.data.DataLoader(
            in_distribution_train, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

        test_images_dataset = torch.utils.data.ConcatDataset(
            [in_distribution_test] + list(out_distribution_datasets.values())
        )
        test_labels = torch.utils.data.TensorDataset(
            torch.cat(
                [torch.zeros(len(in_distribution_test))]  # type: ignore
                + [torch.ones(len(d)) * (i + 1) for i, d in enumerate(out_distribution_datasets.values())]  # type: ignore
            ).long()
        )

        test_dataset = ConcatDatasetsDim1([test_images_dataset, test_labels])
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def benchmark(self, methods: Dict[str, Callable]):
        self._setup()

        for x, y in self.fit_dataloader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            for method in methods.values():
                method.fit(x, y)

        test_labels = []
        test_scores = defaultdict(list)
        for x, y, labels in self.test_dataloader:
            x = x.to(self.device, non_blocking=True)

            test_labels.append(labels)
            for method_name, method in methods.items():
                test_scores[method_name].append(method(x).detach().cpu())

        test_labels = torch.cat(test_labels)
        test_scores = {k: torch.cat(v) for k, v in test_scores.items()}

        res_obj = self.eval(test_scores, test_labels, self.ood_datasets)
        self.report(res_obj)

    def eval(self, test_scores: Dict[str, Tensor], test_labels: Tensor, ood_datasets: List[str]):
        test_scores = {k: v.view(-1) for k, v in test_scores.items()}
        test_labels = test_labels.view(-1)

        in_labels = torch.ones(len(test_labels[test_labels == 0]))

        results = defaultdict(dict)
        for method in test_scores.keys():
            in_scores = test_scores[method][test_labels == 0]

            for i, ood_dataset in enumerate(ood_datasets):
                ood_scores = test_scores[method][test_labels == (i + 1)]
                ood_labels = torch.zeros(len(test_labels[test_labels == (i + 1)]))

                _test_scores = torch.cat([in_scores, ood_scores])
                _test_labels = torch.cat([in_labels, ood_labels])

                fprs, tprs, thrs = metrics.roc(_test_scores, _test_labels)
                precision, recall, _ = metrics.precision_recall_curve(_test_scores, _test_labels)
                precision_out, recall_out, _ = metrics.precision_recall_curve(_test_scores, _test_labels, pos_label=0)
                fpr, tpr, thr = fpr_at_fixed_tpr(fprs, tprs, thrs, 0.95)
                auroc = metrics.auc(fprs, tprs).item()
                aupr_in = metrics.auc(recall, precision, reorder=True).item()
                aupr_out = metrics.auc(recall_out, precision_out, reorder=True).item()

                pos_ratio = torch.mean(_test_labels == 1).item()
                detection_error = compute_detection_error(fpr, tpr, pos_ratio)

                results[method][ood_dataset] = {
                    "fpr_at_0.95_tpr": fpr,
                    "tnr_at_0.95_tpr": 1 - fpr,
                    "detection_error": detection_error,
                    "auroc": auroc,
                    "aupr_in": aupr_in,
                    "aupr_out": aupr_out,
                    "thr": thr,
                }

        return results

    def report(self, res_obj: Dict[str, Dict[str, Dict[str, Any]]]):
        # log results
        print("OOD results:")
        for method, results in res_obj.items():
            print(f"Method: {method}")
            for ood_dataset, res in results.items():
                print(f"\t{ood_dataset}:")
                for metric, val in res.items():
                    print(f"\t\t{METRICS_NAMES_PRETTY[metric]}: {val}")

    def __call__(self, *args: Any, **kwds: Any):
        return super().__call__(*args, **kwds)


class OODCifar10Pipeline(OODPipeline):
    def __init__(self, device, batch_size=128, num_workers=4) -> None:
        super().__init__(
            "cifar10",
            [
                "cifar100",
                "svhn",
                "isun",
                "lsun_c",
                "lsun_r",
                "tiny_imagenet_c",
                "tiny_imagenet_r",
                "textures",
                "places365",
                "english_chars",
            ],
            device,
            batch_size,
            num_workers,
        )
