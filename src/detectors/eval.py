"""Module containing evaluation metrics.
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
    idx = min(idxs)
    return float(fprs[idx]), float(tprs[idx]), float(thresholds[idx])


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


def false_positive_rate(tn, fp, fn, tp):
    return fp / (fp + tn)


def false_negative_rate(tn, fp, fn, tp):
    return fn / (tp + fn)


def true_negative_rate(tn, fp, fn, tp):
    # specificity, selectivity or true negative rate (TNR)
    return tn / (fp + tn)


def precision(tn, fp, fn, tp):
    # precision or positive predictive value (PPV)
    return tp / (tp + fp + 1e-6)


def recall(tn, fp, fn, tp):
    # sensitivity, recall, hit rate, or true positive rate
    return tp / (tp + fn)


def true_positive_rate(tn, fp, fn, tp):
    return recall(tn, fp, fn, tp)


def negative_predictive_value(tn, fp, fn, tp):
    return tn / (tn + fn)


def f1_score(tn, fp, fn, tp):
    return 2 * tp / (2 * tp + fp + fn)


def accuracy_score(tn, fp, fn, tp):
    return (tp + tn) / (tp + tn + fp + fn)


def error_score(tn, fp, fn, tp):
    return 1 - accuracy_score(tn, fp, fn, tp)


def threat_score(tn, fp, fn, tp):
    return tp / (tp + fn + fp)


def bin_accuracy(bin, y_pred, y_true):
    return np.sum(y_pred == y_true) / np.abs(bin[1] - bin[0])


def bin_confidence(bin, y_soft):
    return np.sum(y_soft) / np.abs(bin[1] - bin[0])


def maximum_calibration_error(num_bins, labels_pred, labels_true):
    bins = [(i, i + 1 / num_bins) for i in np.linspace(0, 1, num_bins + 1)][:-1]
    res = []
    for bin in bins:
        a = bin_accuracy(bin, labels_pred, labels_true)
        c = bin_confidence(bin, labels_pred)
        res.append(np.abs(a - c))

    return max(res)


def softmax_brier_score(soft, target):
    if len(soft.shape) == 1:
        soft = soft.reshape(1, -1)

    n_classes = soft.shape[1]
    n_samples = soft.shape[0]
    assert n_samples == target, "The number of targets should be equal to the number of softmax samples."

    # one hot transformation of target vector
    target_vector = one_hot_encode(target, n_classes)
    return np.mean(np.sum((soft - target_vector) ** 2, 1), 0)


def one_hot_encode(targets, n):

    vec = np.zeros((len(targets), n))
    for i, t in enumerate(targets):
        vec[i][t] = 1

    return vec


def expected_calibration_error(num_bins, logits, labels_true):
    bins = [(i, i + 1 / num_bins) for i in np.linspace(0, 1, num_bins + 1)][:-1]
    y_softmax = torch.softmax(logits, 1).numpy()
    labels_pred = torch.argmax(logits, 1).numpy()
    labels_true = labels_true.numpy()
    y_soft = np.max(y_softmax, 1)
    n = len(labels_pred)
    res = 0
    for bin in bins:
        lb = y_soft >= bin[0]
        ub = y_soft < bin[1]
        filt = np.logical_and(lb, ub)
        a = bin_accuracy(bin, labels_pred[filt], labels_true[filt])
        c = bin_confidence(bin, y_soft[filt])
        res += np.abs(a - c) * np.abs(bin[1] - bin[0]) / n

    return res


def get_ood_results(in_scores: Tensor, ood_scores: Tensor) -> Dict[str, float]:
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
    detection_error = compute_detection_error(fpr, tpr, pos_ratio)

    results = {
        "fpr_at_0.95_tpr": fpr,
        "tnr_at_0.95_tpr": 1 - fpr,
        "detection_error": detection_error,
        "auroc": auroc,
        "aupr_in": aupr_in,
        "aupr_out": aupr_out,
        "thr": thr,
    }
    return results
