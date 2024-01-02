import json
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import detectors
from detectors.config import RESULTS_DIR
from detectors.ensemble import ensemble_names, fisher_tau_method, get_combine_p_values_fn, p_value_fn
from drift_cpu import main as drift_cpu_main

_logger = logging.getLogger(__name__)


def main(args):
    combine_fn = get_combine_p_values_fn(args.method)
    args.warmup_size = 10000
    args.num_samples = 10000
    fname = os.path.join(RESULTS_DIR, args.pipeline, "scores.csv")
    # pipeline,model,in_dataset_name,drift_dataset_name,method,method_kwargs,warmup_size,seed,warmup_scores,warmup_labels,warmup_preds,in_scores,in_labels,in_preds,drift_scores,drift_labels,drift_preds,time
    all_scores = pd.read_csv(fname)

    # drop duplicates
    all_scores = all_scores.query("warmup_size == @args.warmup_size")
    all_scores = all_scores.drop_duplicates(
        subset=["pipeline", "model", "method", "method_kwargs", "seed"], keep="last"
    )
    if all_scores.query("model == @args.model and method==@args.method").shape[0] == 0:
        ban_methods = ["argm", "projection"]
        aux = all_scores.query(f"model == '{args.model}' and method != @ensemble_names and method != @ban_methods")
        # sort by method
        aux = aux.sort_values(by="method")
        # transform in_scores from str to tensor
        in_scores = np.concatenate(
            aux["in_scores"].apply(lambda x: np.array(json.loads(x), dtype=np.float32).reshape(-1, 1)).values, axis=1
        )
        in_labels = np.concatenate(
            aux["in_labels"].apply(lambda x: np.array(json.loads(x), dtype=np.float32).reshape(-1, 1)).values, axis=1
        )
        drift_scores = np.concatenate(
            aux["drift_scores"].apply(lambda x: np.array(json.loads(x), dtype=np.float32).reshape(-1, 1)).values, axis=1
        )
        drift_labels = np.concatenate(
            aux["drift_labels"].apply(lambda x: np.array(json.loads(x), dtype=np.float32).reshape(-1, 1)).values, axis=1
        )
        warmup_scores = np.concatenate(
            aux["warmup_scores"].apply(lambda x: np.array(json.loads(x), dtype=np.float32).reshape(-1, 1)).values,
            axis=1,
        )
        # assert all in_labels are the same
        assert np.all(np.isclose(in_labels.mean(1), in_labels[:, 0]))
        assert np.all(np.isclose(drift_labels.mean(1), drift_labels[:, 0]))

        in_scores_p_value = p_value_fn(in_scores, warmup_scores)
        in_scores_tau = combine_fn(in_scores_p_value)
        assert np.isnan(in_scores_p_value).sum() == 0
        drift_scores_p_value = p_value_fn(drift_scores, warmup_scores)
        drift_scores_tau = combine_fn(drift_scores_p_value)
        assert np.isnan(drift_scores_p_value).sum() == 0
        warmup_scores_p_value = p_value_fn(warmup_scores, warmup_scores)
        warmup_scores_tau = combine_fn(warmup_scores_p_value)
        assert np.isnan(warmup_scores_p_value).sum() == 0

        # drop copumns from aux
        aux = aux.drop(columns=["method", "in_scores", "drift_scores", "warmup_scores"])
        # print(aux)
        elem = {}
        for c in aux.columns:
            elem[c] = aux[c].values[0]
        elem["method"] = args.method
        elem["in_scores"] = json.dumps(in_scores_tau.tolist())
        elem["drift_scores"] = json.dumps(drift_scores_tau.tolist())
        elem["warmup_scores"] = json.dumps(warmup_scores_tau.tolist())
        # reorder elem columns
        elem = pd.DataFrame(elem, index=[0])
        elem = elem[all_scores.columns]
        elem_dict = elem.to_dict(orient="records")[0]

        # insert elem to all_scores
        # all_scores = pd.concat([all_scores, pd.DataFrame(elem, index=[0])], ignore_index=True)

        # save results to file
        detectors.utils.append_results_to_csv_file(elem_dict, fname)

    # run drift_cpu.py
    # args.method = "ensemble"
    args.method_kwargs = {}
    _logger.info("Done ensembling scores")
    _logger.info("Running drift_cpu.py")
    drift_cpu_main(args)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=str, default="drift_benchmark_imagenet_r")
    parser.add_argument("--method", type=str, default="fisher")
    parser.add_argument("--model", type=str, default="resnet50.tv_in1k")
    parser.add_argument("--criterion", type=str, default="ks_2samp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
