import argparse
import json
import logging
import os

import torch

import detectors
from detectors.config import RESULTS_DIR
from detectors.utils import str_to_dict

_logger = logging.getLogger(__name__)


def main(args):
    print(f"Running {args.pipeline} pipeline on {args.model} model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dataset = args.pipeline.split("_")[-1]
    model = detectors.create_model(args.model, pretrained=True)
    model.to(device)
    test_transform = detectors.create_transform(model)
    method_kwargs = args.method_kwargs or {}
    method = detectors.create_detector(args.method, model=model, **method_kwargs)
    hyperparameters = detectors.create_hyperparameters(args.method)
    best_kwargs = {}
    n_trials = args.n_trials * len(hyperparameters)
    if len(hyperparameters) > 0:
        pipeline = detectors.create_pipeline(
            args.pipeline,
            batch_size=args.batch_size,
            seed=args.seed,
            transform=test_transform,
            limit_fit=args.limit_fit,
            limit_run=args.limit_run,
        )
        pipeline_results = pipeline.run(
            method,
            hyperparameters=hyperparameters,
            objective_metric=args.objective,
            objective_dataset=args.objective_dataset,
            n_trials=n_trials,
        )
        print(pipeline.report(pipeline_results))
        best_kwargs = pipeline_results["best_params"]

    method_kwargs.update(best_kwargs)
    method = detectors.create_detector(args.method, model=model, **method_kwargs)

    test_pipeline = detectors.create_pipeline(
        f"ood_benchmark_{in_dataset}",
        batch_size=args.batch_size,
        seed=args.seed,
        transform=test_transform,
        limit_fit=args.limit_fit,
        limit_run=1,
    )
    pipeline_results = test_pipeline.run(method)
    print(test_pipeline.report(pipeline_results))
    results = {
        "model": args.model,
        "method": args.method,
        "best_kwargs": method_kwargs,
        "n_trials": n_trials,
        "seed": args.seed,
        "limit_fit": args.limit_fit,
        "limit_run": args.limit_run,
        "objective": args.objective,
        "objective_dataset": args.objective_dataset,
        **pipeline_results["results"],
    }
    filename = os.path.join(RESULTS_DIR, args.pipeline, "results.csv")
    detectors.utils.append_results_to_csv_file(results, filename)
    print("Results saved to {}".format(filename))

    scores = pipeline_results["scores"]
    labels = pipeline_results["labels"]

    results = {
        "model": args.model,
        "in_dataset_name": test_pipeline.in_dataset_name,
        "out_datasets_names": test_pipeline.out_datasets_names,
        "method": args.method,
        "method_kwargs": method_kwargs,
        "scores": scores.numpy().tolist(),
        "labels": labels.numpy().tolist(),
    }
    filename = os.path.join(RESULTS_DIR, args.pipeline, "scores.csv")
    detectors.utils.append_results_to_csv_file(results, filename)
    print("Scores saved to {}".format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="odin")
    parser.add_argument("--method_kwargs", type=str_to_dict, default="{}", help="")
    parser.add_argument("--pipeline", type=str, default="ood_validation_cifar10")
    parser.add_argument("--model", type=str, default="resnet18_cifar10")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--limit_fit", type=float, default=1)
    parser.add_argument("--limit_run", type=float, default=0.1)
    parser.add_argument("--objective", type=str, default="auroc")
    parser.add_argument("--objective_dataset", type=str, default="average")
    parser.add_argument("--n_trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _logger.info(json.dumps(args.__dict__, indent=2))

    main(args)
