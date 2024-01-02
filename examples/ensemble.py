import copy
import json
import multiprocessing as mp
import os
import pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import sklearn
import sklearn.preprocessing
import torch
from tqdm import tqdm

METHODS = [
    "msp",
    "energy",
    "kl_matching",
    "igeood_logits",
    "max_logits",
    # "react",
    # "dice",
    # "gmm",
    # "vim",
    # "gradnorm",
    "mahalanobis",
    "relative_mahalanobis",
    "knn_euclides",
    "knn_cosine",
    # "knn_projection",
    "maxcosine",
    "odin",
    "doctor",
    # "mcdropout",
    # "projection",
]


def score_to_probability(score):
    return 1 / (1 + np.exp(-score))


def probability_to_score(probability):
    return np.log(probability / (1 - probability))


def score_to_normal_cdf(score):
    return scipy.stats.norm.cdf(score)


def preprocess(x, score, mode="minmax", **kwargs):
    x_0 = x.copy()
    if mode == "minmax":
        qt = sklearn.preprocessing.MinMaxScaler().fit(score)
        x_0 = qt.transform(x)
    elif mode == "minmax_center":
        qt = sklearn.preprocessing.MinMaxScaler().fit(score)
        x_0 = qt.transform(x)
        x_0 = x_0 - np.mean(x_0, axis=0)
    elif mode == "standard":
        x_0 = (x - np.mean(score, axis=0, keepdims=True)) / np.std(score, axis=0, keepdims=True)
    elif mode == "quantile":
        qt = sklearn.preprocessing.QuantileTransformer(n_quantiles=1000, output_distribution="normal").fit(score)
        x_0 = qt.transform(x)
    elif mode == "robust":
        qt = sklearn.preprocessing.RobustScaler().fit(score)
        x_0 = qt.transform(x)
    else:
        raise NotImplementedError

    return x_0


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def initialize_weights(k, batch_size, mode="uniform"):
    if mode == "uniform":
        w = np.ones((batch_size, k))
    elif mode == "random":
        w = np.random.rand((batch_size, k))
    else:
        raise NotImplementedError

    return softmax(w)


def kl_divergence(p, q, eps=1e-10):
    return np.sum(p * np.log(p / q + eps), axis=1)


def pairwise_kl_divergence(p, q, eps=1e-12):
    """Pairwise KL divergence between two distributions.

    Args:
        p (ndarray): dimensions M x K
        q (ndarray): dimensions M x K
        eps (float, optional): Defaults to 1e-10.

    Returns:
        ndarray: dimensions M x K x K
    """
    kl = np.log(p[:, :, None] / (q[:, None, :] + eps) + eps)
    # make sure that diagonal is zero for every sub matrix
    for i in range(kl.shape[0]):
        np.fill_diagonal(kl[i], 0)
    return kl


def mutual_information_omega__z_hat(w, p):
    """Mutual information between the weights of the ensemble and the predictions of the ensemble.

    Args:
        w (ndarray): dimensions M x K
        p (ndarray): M x K

    Returns:
        ndarray: M
    """
    assert w.shape == p.shape
    return np.min(np.sum(w[:, None, :] * pairwise_kl_divergence(p, p), axis=-1), axis=-1)


def format_weights(w, k, batch_size=1):
    w = w.reshape((batch_size, k))
    w = softmax(w)
    return w


def objective_function(w, preprocessed, k, batch_size=1):
    w = format_weights(w, k, batch_size)
    return -mutual_information_omega__z_hat(w, preprocessed).mean()


def get_key():
    return ["method", "seed", "model", "method_kwargs", "corruption"]


def main():
    pipeline_names = ["covariate_drift_cifar10", "covariate_drift_cifar100"]
    pipeline_name = "covariate_drift_cifar10"
    fileversion = "v4"

    for i, pipeline_name in enumerate(pipeline_names):
        print(pipeline_name)
        df1 = pd.read_csv(f"results/{pipeline_name}/results_{fileversion}.csv")
        # DROP DUPLICATES
        df1 = df1.drop_duplicates(subset=get_key(), keep="last")
        df2 = pd.read_csv(f"results/{pipeline_name}/scores_{fileversion}.csv")
        # DROP DUPLICATES
        df2 = df2.drop_duplicates(subset=get_key(), keep="last")
        assert df1.shape[0] == df2.shape[0]

        if i == 0:
            df = pd.merge(df1, df2, on=get_key(), how="inner")
        else:
            tmp = pd.merge(df1, df2, on=get_key(), how="inner")
            df = pd.concat([df, tmp], axis=0).reset_index(drop=True)

    df = df.query("method in @METHODS")
    df = df.dropna()

    for k in ["scores", "preds", "labels", "drift_labels", "moving_average", "method_kwargs", "targets", "mistakes"]:
        df[k] = df[k].apply(lambda x: json.loads(x))
    df = df.query("seed == 1")

    data = {}
    for model in df["model"].unique():
        for corruption in df["corruption"].unique():
            sub = df.query(f"model == '{model}' and corruption == '{corruption}'")
            # sub = sub[["model", "method", "scores", "labels", "preds", "targets", "" "corruption"]]
            # group by method and extract scors to a numpy array
            sub = sub.groupby(["model", "method", "corruption"])

            scores = np.concatenate(sub["scores"].apply(lambda x: np.array(list(x))).reset_index()["scores"].values)
            labels = np.concatenate(sub["labels"].apply(lambda x: np.array(list(x))).reset_index()["labels"].values)
            preds = np.concatenate(sub["preds"].apply(lambda x: np.array(list(x))).reset_index()["preds"].values)
            targets = np.concatenate(sub["targets"].apply(lambda x: np.array(list(x))).reset_index()["targets"].values)
            mistakes = np.concatenate(
                sub["mistakes"].apply(lambda x: np.array(list(x))).reset_index()["mistakes"].values
            )
            drift_labels = np.concatenate(
                sub["drift_labels"].apply(lambda x: np.array(list(x))).reset_index()["drift_labels"].values
            )
            moving_average = np.concatenate(
                sub["moving_average"].apply(lambda x: np.array(list(x))).reset_index()["moving_average"].values
            )

            data[(model, corruption)] = {
                "scores": scores,
                "labels": labels[0, :],
                "preds": preds[0, :],
                "targets": targets[0, :],
                "drift_labels": drift_labels[0, :],
                "mistakes": mistakes[0, :],
                "moving_average": moving_average,
                "methods": [i[1] for i in list(sub.groups.keys())],
                "corruption": corruption,
                "model": model,
            }

    for k in tqdm(data.keys()):
        scores = data[k]["scores"].T
        moving_average = data[k]["moving_average"].T
        # preprocess scores
        warmup_size = 2000
        test_size = 1666
        limit = warmup_size + test_size / 2

        scores = score_to_probability(preprocess(scores, scores[0:limit, :], mode="quantile"))
        moving_average = score_to_probability(preprocess(moving_average, moving_average[0:limit, :], mode="quantile"))
        data[k]["scores_preprocessed"] = scores
        data[k]["moving_average_preprocessed"] = moving_average

    return data


def worker(config, data, shared_data):
    scores = data[config]["scores_preprocessed"]
    moving_average = data[config]["moving_average_preprocessed"]
    w0 = initialize_weights(scores.shape[1], 1, mode="uniform")
    mutual_scores = np.zeros(scores.shape[0])
    mutual_moving_average = np.zeros(scores.shape[0])
    for i in range(scores.shape[0]):
        preprocessed = scores[i, :].reshape((1, scores.shape[1]))

        def objective_function(w):
            w = format_weights(w, scores.shape[1], batch_size=1)
            return -mutual_information_omega__z_hat(w, preprocessed).mean()

        results = scipy.optimize.minimize(fun=objective_function, x0=w0.flatten(), method="SLSQP")
        mutual_scores[i] = np.sum(format_weights(results.x, scores.shape[1], 1) * preprocessed)

        # moving average
        preprocessed = moving_average[i, :].reshape((1, moving_average.shape[1]))
        results = scipy.optimize.minimize(fun=objective_function, x0=w0.flatten(), method="SLSQP")
        mutual_moving_average[i] = np.sum(format_weights(results.x, scores.shape[1], 1) * preprocessed)

    shared_data[config] = {"mutual_scores": mutual_scores, "mutual_moving_average": mutual_moving_average}


def parallel_main():
    data = main()

    with open("results/covariate_drift_data.pkl", "wb") as f:
        pickle.dump(data, f)

    # multiprocessing data dict
    configs = list(data.keys())

    manager = mp.Manager()
    shared_data = manager.dict()
    jobs = []

    # change to a Pool of process
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        with tqdm(total=len(configs)) as pbar:
            for _ in pool.imap_unordered(partial(worker, data=data, shared_data=shared_data), configs):
                pbar.update()

    print(shared_data[list(shared_data.keys())[0]])
    for k in shared_data.keys():
        data[k] = {**data[k], **shared_data[k]}

    # save data
    with open("results/covariate_drift_mutual_information.pkl", "wb") as f:
        pickle.dump(dict(data), f)


if __name__ == "__main__":
    main()
