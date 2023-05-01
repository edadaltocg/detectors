import argparse
import logging
import os
from typing import Any, Dict

import timm
import timm.data
import torch

import detectors
from detectors.config import RESULTS_DIR
from detectors.data.constants import CORRUPTIONS
from detectors.utils import append_results_to_csv_file, str_to_dict

_logger = logging.getLogger(__name__)


def main(
    model_name: str,
    method_name: str,
    method_kwargs: Dict[str, Any] = {},
    pipeline_name="covariate_drift_cifar10",
    corruption=None,
    intensities=[1, 2, 3, 4, 5],
    batch_size=128,
    limit_fit=1.0,
    subsample=1,
    seed=42,
    debug=False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_name, pretrained=True)
    data_config = timm.data.resolve_data_config(model.default_cfg)
    transform = timm.data.create_transform(**data_config)
    model = model.to(device)
    model.eval()
    pipeline = detectors.create_pipeline(
        pipeline_name,
        corruption=corruption,
        intensities=intensities,
        batch_size=batch_size,
        limit_fit=limit_fit,
        transform=transform,
        seed=seed,
    )
    _logger.info("Method kwargs: %s", method_kwargs)

    method = detectors.create_detector(method_name, model=model, **method_kwargs)
    # fit detector
    method = pipeline.preprocess(method)
    # run benchmark pipeline
    results = pipeline.run(method, model)

    if not debug:
        pipeline.report(results, subsample=subsample)

        # save results to csv file
        path = os.path.join(RESULTS_DIR, pipeline_name, "results.csv")
        save_results = {
            "model": model_name,
            "method": method_name,
            "method_kwargs": method_kwargs,
            "corruption": corruption,
            "intensities": intensities,
            "corr_acc": results["corr_acc"],
            "corr_drift": results["corr_drift"],
            "auroc_drift": results["auroc_drift"],
            "auroc_mistakes": results["auroc_mistakes"],
            "fpr_drift": results["fpr_drift"],
            "fpr_mistakes": results["fpr_mistakes"],
            "first_drift": results["first_drift"],
            "splits": results["splits"],
            "seed": seed,
        }
        append_results_to_csv_file(save_results, path)

        scores = results["scores"]
        labels = results["labels"]
        preds = results["preds"]
        targets = results["targets"]

        results = {
            "model": model_name,
            "method": method_name,
            "method_kwargs": method_kwargs,
            "scores": scores.numpy().tolist(),
            "labels": labels.numpy().tolist(),
            "preds": preds.numpy().tolist(),
            "targets": targets.numpy().tolist(),
        }
        filename = os.path.join(RESULTS_DIR, args.pipeline, "scores.csv")
        append_results_to_csv_file(results, filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18_cifar10")
    parser.add_argument("--method", type=str, default="msp")
    parser.add_argument("--method_kwargs", type=str_to_dict, default={}, help='{"temperature":1000, "eps":0.00014}')
    parser.add_argument("--pipeline", type=str, default="covariate_drift_cifar10")
    parser.add_argument("-idx", "--corruption_idx", type=int, default=0)
    parser.add_argument("--intensities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--limit_fit", type=float, default=0.5)
    parser.add_argument("--subsample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    main(
        model_name=args.model,
        method_name=args.method,
        method_kwargs=args.method_kwargs,
        pipeline_name=args.pipeline,
        corruption=CORRUPTIONS[args.corruption_idx],
        intensities=args.intensities,
        batch_size=args.batch_size,
        limit_fit=args.limit_fit,
        subsample=args.subsample,
        seed=args.seed,
        debug=args.debug,
    )
