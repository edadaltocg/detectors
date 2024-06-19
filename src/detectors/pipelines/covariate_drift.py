import logging
from typing import Any, Dict, List

import accelerate
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.metrics
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import Tensor
from tqdm import tqdm

import detectors
from detectors.data import create_dataset
from detectors.methods import DetectorWrapper
from detectors.pipelines import register_pipeline
from detectors.pipelines.base import Pipeline
from detectors.utils import ConcatDatasetsDim1

_logger = logging.getLogger(__name__)


class CovariateDriftPipeline(Pipeline):
    def __init__(
        self,
        dataset_name: str,
        dataset_splits: List[str],
        transform,
        corruptions: List[str],
        intensities: List[int],
        batch_size: int = 128,
        limit_fit: float = 1.0,
        warmup_size=2000,
        seed=42,
        num_workers: int = 3,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        **kwargs,
    ) -> None:
        """Covariate Drift Pipeline.

        Covariate drift event: when moving accuracy is below threshold compared to training accuracy

        Args:
            dataset_name (str): Name of the dataset.
            dataset_splits (List[str]): List of dataset splits to use in fit and in dataset, respectively.
            transform (Callable): Transform to apply to the dataset.
            corruptions (List[str]): List of corruptions to apply.
            intensities (List[int]): List of intensities to apply.
            batch_size (int, optional): Batch size. Defaults to 128.

        """
        self.accelerator = accelerate.Accelerator()
        accelerate.utils.set_seed(seed)

        _logger.info("Creating datasets...")

        fit_dataset = create_dataset(
            dataset_name, split=dataset_splits[0], transform=transform
        )
        # shuffle fit dataset
        limit_fit = limit_fit or 1
        limit_fit = min(int(limit_fit * len(fit_dataset)), len(fit_dataset))
        indices = torch.randperm(len(fit_dataset)).numpy()[:limit_fit]
        fit_dataset = torch.utils.data.Subset(fit_dataset, indices)
        in_dataset = create_dataset(
            dataset_name, split=dataset_splits[1], transform=transform
        )
        # shuffle in dataset
        indices = torch.randperm(len(in_dataset)).numpy()
        max_dataset_size = len(in_dataset) // (len(corruptions) * len(intensities) + 1)
        self.splits = torch.arange(0, len(in_dataset), max_dataset_size)
        in_dataset = torch.utils.data.Subset(
            in_dataset, indices[np.arange(0, self.splits[1].item())]
        )
        warmup_dataset = torch.utils.data.Subset(
            fit_dataset, range(len(fit_dataset))[:warmup_size]
        )
        in_dataset = torch.utils.data.ConcatDataset([warmup_dataset, in_dataset])
        out_datasets = {}
        for i, corruption in enumerate(corruptions):
            out_datasets[corruption] = []
            for j, intensity in enumerate(intensities):
                _indices = indices[
                    torch.arange(
                        self.splits[i + j + 1].item(), self.splits[i + j + 2].item()
                    )
                ]
                out_datasets[corruption].append(
                    torch.utils.data.Subset(
                        create_dataset(
                            dataset_name + "_c",
                            split=corruption,
                            intensity=intensity,
                            transform=transform,
                        ),
                        _indices,
                    )
                )
        self.splits = self.splits.numpy() + warmup_size
        self.splits = [0] + self.splits.tolist()
        _logger.info("Data splits are: %s", self.splits)

        # increasing intensity concat dataset
        out_dataset = []
        for i, intensity in enumerate(intensities):
            # shuffle corruptions?
            out_dataset.append(
                torch.utils.data.ConcatDataset(
                    [out_datasets[corruption][i] for corruption in corruptions]
                )
            )
        out_dataset = torch.utils.data.ConcatDataset(out_dataset)

        _logger.debug("Fit dataset size: %s", {len(fit_dataset)})
        _logger.debug("In dataset size: %s", {len(in_dataset)})
        _logger.debug("Out dataset size: %s", {len(out_dataset)})

        self.in_dataset = in_dataset
        self.out_dataset = out_dataset
        self.fit_dataset = fit_dataset
        self.batch_size = batch_size
        self.warmup_size = warmup_size
        self.limit_fit = limit_fit
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        self.setup()

    def setup(self, *args):
        test_dataset = torch.utils.data.ConcatDataset(
            [self.in_dataset, self.out_dataset]
        )
        test_labels = torch.utils.data.TensorDataset(
            torch.cat(
                [torch.zeros(len(self.in_dataset))]
                + [torch.ones(len(self.out_dataset))]
            ).long()  # type: ignore
        )

        self.test_dataset = ConcatDatasetsDim1([test_dataset, test_labels])
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )
        self.fit_dataloader = torch.utils.data.DataLoader(
            self.fit_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )
        self.fit_dataloader = self.accelerator.prepare(
            self.fit_dataloader
        )  # careful with this with multiple gpus
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

    def preprocess(self, method: DetectorWrapper) -> DetectorWrapper:
        if not hasattr(method.detector, "update"):
            return method

        if self.fit_dataset is None:
            _logger.warning("Fit is not set or not supported. Returning.")
            return method

        if method.model is not None:
            method.detector.model = self.accelerator.prepare(method.detector.model)

        progress_bar = tqdm(
            range(len(self.fit_dataloader)),
            desc="Fitting",
            disable=not self.accelerator.is_local_main_process,
        )
        method.start()
        for x, y in self.fit_dataloader:
            method.update(x, y)
            progress_bar.update(1)
        progress_bar.close()
        self.accelerator.wait_for_everyone()
        method.end()
        return method

    def run(self, method, model, **kwargs):
        method = method

        test_labels = torch.empty(len(self.test_dataset), dtype=torch.long)
        test_targets = torch.empty(len(self.test_dataset), dtype=torch.long)
        test_scores = torch.empty(len(self.test_dataset), dtype=torch.float)
        test_preds = torch.empty(len(self.test_dataset), dtype=torch.long)
        idx = 0
        progress_bar = tqdm(
            range(len(self.test_dataloader)),
            desc="Inference",
            disable=not self.accelerator.is_local_main_process,
        )
        for x, y, labels in self.test_dataloader:
            scores = method(x)
            with torch.no_grad():
                logits = model(x)

            labels, y, scores, logits = self.accelerator.gather_for_metrics(
                (labels, y, scores, logits)
            )

            test_labels[idx : idx + len(x)] = labels.detach().cpu()
            test_targets[idx : idx + len(x)] = y.detach().cpu()
            test_scores[idx : idx + len(x)] = scores.detach().cpu()
            test_preds[idx : idx + len(x)] = logits.detach().cpu().argmax(1)

            idx += len(x)

            progress_bar.update(1)
        progress_bar.close()
        self.accelerator.wait_for_everyone()

        assert (
            len(test_labels)
            == len(test_targets)
            == len(test_scores)
            == len(test_preds)
            == idx
        )

        _logger.info("Computing metrics...")
        acc_threshold = kwargs.get("acc_threshold", 0.90)
        stride = kwargs.get("stride", 1)
        metrics = self.postprocess(
            test_scores,
            test_preds,
            test_targets,
            test_labels,
            self.batch_size,
            self.warmup_size,
            self.splits,
            acc_threshold=acc_threshold,
            stride=stride,
        )

        return {"method": method, **metrics}

    @staticmethod
    def postprocess(
        test_scores: Tensor,
        test_preds: Tensor,
        test_targets: Tensor,
        test_labels: Tensor,
        win_size: int,
        warmup_size: int,
        splits: List,
        stride=1,
        alpha=0.99,
        acc_threshold=0.90,
        moving_average=None,
        **kwargs,
    ) -> Dict[str, Any]:
        avg_warmup = test_scores[:warmup_size].mean().item()
        data_padded = F.pad(
            test_scores.unsqueeze(0), (win_size - 1, 0), "constant", avg_warmup
        ).squeeze(0)
        if moving_average is None:
            moving_average = data_padded.unfold(0, win_size, stride).mean(dim=1)

        ema = test_scores.clone()
        ema[0] = avg_warmup
        for i in range(1, len(test_scores)):
            ema[i] = alpha * ema[i - 1] + (1 - alpha) * test_scores[i]

        mistakes = (test_preds != test_targets).float()
        mistakes_padded = F.pad(
            mistakes.unsqueeze(0), (win_size - 1, 0), "constant", 0
        ).squeeze(0)
        moving_accuracy = 1 - mistakes_padded.unfold(0, win_size, stride).mean(dim=1)

        # define real drift event: when moving accuracy is below threshold compared to training accuracy
        acc = (
            moving_accuracy[splits[1] : splits[1] + (splits[2] - splits[1]) // 2]
            .mean()
            .item()
        )
        ref = acc_threshold * acc
        _logger.info("Original accuracy: %s", acc)
        _logger.info("Reference accuracy to detect drift: %s", ref)

        drift_labels = (moving_accuracy < ref).float()
        corr_drift = np.corrcoef(-moving_average.numpy(), drift_labels.numpy())[0, 1]
        corr_acc = np.corrcoef(moving_average.numpy(), moving_accuracy.numpy())[0, 1]

        # check error if theres is only one label on drift_labels
        if len(np.unique(drift_labels.numpy())) == 1:
            auroc_drift = 1.0
        else:
            auroc_drift = float(
                sklearn.metrics.roc_auc_score(drift_labels, -moving_average)
            )

        fprs, tprs, thresholds = sklearn.metrics.roc_curve(
            drift_labels, -moving_average
        )
        fpr_drift, _, _ = detectors.eval.fpr_at_fixed_tpr(fprs, tprs, thresholds, 0.95)

        if len(np.unique(mistakes.numpy())) == 1:
            auroc_mistakes = 1.0
        else:
            auroc_mistakes = float(
                sklearn.metrics.roc_auc_score(mistakes, -test_scores)
            )

        fprs, tprs, thresholds = sklearn.metrics.roc_curve(mistakes, -test_scores)
        fpr_mistakes, _, _ = detectors.eval.fpr_at_fixed_tpr(
            fprs, tprs, thresholds, 0.95
        )

        return dict(
            scores=test_scores,
            preds=test_preds,
            targets=test_targets,
            labels=test_labels,
            drift_labels=drift_labels,
            first_drift=torch.argmax(drift_labels).item(),
            ref_accuracy=ref,
            splits=splits,
            corr_acc=corr_acc,
            corr_drift=corr_drift,
            auroc_drift=auroc_drift,
            auroc_mistakes=auroc_mistakes,
            fpr_drift=fpr_drift,
            fpr_mistakes=fpr_mistakes,
            moving_average=moving_average,
            ema=ema,
            moving_accuracy=moving_accuracy,
            mistakes=mistakes,
            window_size=win_size,
        )

    @staticmethod
    def report(results: Dict[str, Any], subsample=10, warmup_size=2000, **kwargs):
        print("Results:")
        print("\tCorr. Acc:", results["corr_acc"])
        print("\tCorr. Drift:", results["corr_drift"])
        print("\tAUC Drift:", results["auroc_drift"])
        print("\tAUC Mistakes:", results["auroc_mistakes"])
        print("\tFPR Drift:", results["fpr_drift"])
        print("\tFPR Mistakes:", results["fpr_mistakes"])
        print("\tFirst Drift:", results["first_drift"])
        print("\tSplits", results["splits"])

        # plot results
        mpl_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 4))
        ax2 = ax1.twinx()
        # subsample values
        test_scores = results["scores"][::subsample]
        moving_average = results["moving_average"][::subsample]
        ema = results["ema"][::subsample]
        # test_labels = results["test_labels"][::subsample]
        drift_labels = results["drift_labels"][::subsample]
        mistakes = results["mistakes"][::subsample]
        moving_accuracy = results["moving_accuracy"][::subsample]

        ax1.plot(test_scores.numpy(), alpha=0.4, c=mpl_colors[1], label="score")
        ax1.plot(moving_average.numpy(), alpha=0.8, c=mpl_colors[0], label="moving avg")
        ax1.plot(ema.numpy(), alpha=0.8, c=mpl_colors[4], linewidth=2, label="ema")
        ax1.grid()

        ax2.vlines(
            warmup_size // subsample,
            0,
            1,
            linestyle="--",
            color="gray",
            alpha=0.5,
            linewidth=3,
            label="begin test set",
        )
        ax2.plot(
            drift_labels.numpy(),
            alpha=0.5,
            c=mpl_colors[2],
            linestyle="--",
            label="drift",
            linewidth=3,
        )
        ax2.scatter(
            range(len(mistakes)),
            mistakes.numpy(),
            alpha=0.5,
            marker="*",
            c=mpl_colors[3],
            label="mistakes",
        )
        ax2.plot(
            moving_accuracy.numpy(),
            alpha=0.5,
            c=mpl_colors[3],
            label="moving accuracy",
            linewidth=2,
        )
        # plot reference accuracy
        ax2.axhline(
            results["ref_accuracy"],
            linestyle=":",
            color="black",
            alpha=0.5,
            linewidth=3,
            label="drift accuracy ref",
        )
        ax1.set_xlabel("Sample index")
        ax1.set_ylabel("Scores")
        ax2.set_ylabel("Drift")
        ax2.legend(loc="upper right")
        ax1.legend(loc="lower left")

        plt.suptitle(
            f"Corr. Acc. {results['corr_acc']:.2f}\nFPR Mistakes {results['fpr_mistakes']:.2f}, AUC Mistakes {results['auroc_mistakes']:.2f}, AUC Drift {results['auroc_drift']:.2f}"
        )


@register_pipeline("covariate_drift_cifar10")
class OneCorruptionCovariateDriftCifar10Pipeline(CovariateDriftPipeline):
    def __init__(
        self,
        transform,
        corruption: str,
        intensities: List[int],
        batch_size: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(
            "cifar10",
            ["train", "test"],
            transform,
            [corruption],
            intensities,
            batch_size,
            **kwargs,
        )


@register_pipeline("covariate_drift_cifar100")
class OneCorruptionCovariateDriftCifar100Pipeline(CovariateDriftPipeline):
    def __init__(
        self,
        transform,
        corruption: str,
        intensities: List[int],
        batch_size: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(
            "cifar100",
            ["train", "test"],
            transform,
            [corruption],
            intensities,
            batch_size,
            **kwargs,
        )


@register_pipeline("covariate_drift_imagenet")
class OneCorruptionCovariateDriftImagenetPipeline(CovariateDriftPipeline):
    def __init__(
        self,
        transform,
        corruption: str,
        intensities: List[int],
        batch_size: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(
            "imagenet",
            ["train", "val"],
            transform,
            [corruption],
            intensities,
            batch_size,
            **kwargs,
        )
