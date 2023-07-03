"""
Module containing evaluation metrics.
"""
from typing import Dict

import numpy as np
import sklearn
import sklearn.metrics
import torch
from torch import Tensor


def fpr_at_fixed_tpr(fprs: np.ndarray, tprs: np.ndarray, thresholds: np.ndarray, tpr_level: float = 0.95):
    """Return the FPR at a fixed TPR level.

    Args:
        fprs (np.ndarray): False positive rates.
        tprs (np.ndarray): True positive rates.
        thresholds (np.ndarray): Thresholds.
        tpr_level (float, optional): TPR level. Defaults to 0.95.

    Returns:
        Tuple[float, float, float]: FPR, TPR, threshold.
    """
    # return np.interp(tpr_level, tprs, fprs)
    if all(tprs < tpr_level):
        raise ValueError(f"No threshold allows for TPR at least {tpr_level}.")
    idxs = [i for i, x in enumerate(tprs) if x >= tpr_level]
    if len(idxs) == 0:
        idx = 0
    else:
        idx = min(idxs)
    return float(fprs[idx]), float(tprs[idx]), float(thresholds[idx])


def fnr_at_fixed_tnr(fprs: np.ndarray, tprs: np.ndarray, thresholds: np.ndarray, tnr_level: float = 0.95):
    """Return the FNR at a fixed TNR level.

    Args:
        fprs (np.ndarray): False positive rates.
        tprs (np.ndarray): True positive rates.
        thresholds (np.ndarray): Thresholds.
        tnr_level (float, optional): TNR level. Defaults to 0.95.

    Returns:
        Tuple[float, float, float]: FNR, TNR, threshold."""
    tnrs = 1 - fprs
    fnrs = 1 - tprs

    if all(tnrs < tnr_level):
        raise ValueError(f"No threshold allows for TNR at least {tnr_level}.")
    idxs = [i for i, x in enumerate(tnrs) if x >= tnr_level]
    idx = min(idxs)
    return float(fnrs[idx]), float(tnrs[idx]), float(thresholds[idx])


def compute_detection_error(fpr: float, tpr: float, pos_ratio: float):
    """Compute the detection error.

    Args:
        fpr (float): False positive rate at a fixed TPR.
        tpr (float): True positive rate.
        pos_ratio (float): Ratio of positive labels.

    Returns:
        float: Detection error.
    """
    # Get ratios of positives to negatives
    neg_ratio = 1 - pos_ratio
    # Get indexes of all TPR >= fixed tpr level
    detection_error = pos_ratio * (1 - tpr) + neg_ratio * fpr
    return detection_error


def minimum_detection_error(fprs: np.ndarray, tprs: np.ndarray, pos_ratio: float):
    """Compute the minimum detection error.

    Args:
        fprs (np.ndarray): False positive rates.
        tprs (np.ndarray): True positive rates.
        thresholds (np.ndarray): Thresholds.
        pos_ratio (float): Ratio of positive labels.

    Returns:
        Tuple[float, float, float]: FPR, TPR, threshold.
    """
    detection_errors = [compute_detection_error(fpr, tpr, pos_ratio) for fpr, tpr in zip(fprs, tprs)]
    idx = np.argmin(detection_errors)
    return detection_errors[idx]


def aufnr_aufpr_autc(fprs: np.ndarray, tprs: np.ndarray, thresholds: np.ndarray):
    """Compute the AUFNR, AUFPR, and AUTC metrics.

    Args:
        fprs (np.ndarray): False positive rates.
        tprs (np.ndarray): True positive rates.
        thresholds (np.ndarray): Thresholds.


    Returns:
        Tuple[float, float, float]: AUFNR, AUFPR, AUTC.

    References:
        [1] Humblot-Renaux et. al. Beyond AUROC \& Co. for Evaluating Out-of-Distribution Detection Performance. 2023.
    """
    fnrs = 1 - tprs
    sorted_idx = np.argsort(thresholds)
    aufnr = sklearn.metrics.auc(thresholds[sorted_idx], fnrs[sorted_idx])
    aufpr = sklearn.metrics.auc(thresholds[sorted_idx], fprs[sorted_idx])
    autc = (aufnr + aufpr) / 2
    return float(aufnr), float(aufpr), float(autc)


def get_ood_results(in_scores: Tensor, ood_scores: Tensor) -> Dict[str, float]:
    """Compute OOD detection metrics.

    Args:
        in_scores (Tensor): In-distribution scores.
        ood_scores (Tensor): Out-of-distribution scores.

    Returns:
        Dict[str, float]: OOD detection metrics.
            keys: `fpr_at_0.95_tpr`, `tnr_at_0.95_tpr`, `detection_error`, `auroc`, `aupr_in`, `aupr_out`, `thr`.
    """
    if isinstance(in_scores, np.ndarray) or isinstance(in_scores, list):
        in_scores = torch.tensor(in_scores)
    if isinstance(ood_scores, np.ndarray) or isinstance(ood_scores, list):
        ood_scores = torch.tensor(ood_scores)
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
    detection_error = minimum_detection_error(fprs, tprs, pos_ratio)

    aufnr, aufpr, autc = aufnr_aufpr_autc(fprs, tprs, thrs)

    results = {
        "fpr_at_0.95_tpr": fpr,
        "tnr_at_0.95_tpr": 1 - fpr,
        "detection_error": detection_error,
        "auroc": auroc,
        "aupr_in": aupr_in,
        "aupr_out": aupr_out,
        "aufnr": aufnr,
        "aufpr": aufpr,
        "autc": autc,
        "thr": thr,
    }
    return results
