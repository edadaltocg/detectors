import argparse
import json
import logging
import os
import sys

import timm
import timm.data
import torch

import detectors
from detectors.config import RESULTS_DIR

_logger = logging.getLogger(__name__)


def main(args):
    print(f"Running {args.pipeline} pipeline on {args.model} model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create model
    model = timm.create_model(args.model, pretrained=True)
    model.to(device)
    data_config = timm.data.resolve_data_config(model.default_cfg)
    test_transform = timm.data.create_transform(**data_config)
    _logger.info("Test transform: %s", test_transform)
    x = torch.randn(1, 3, 224, 224).to(device)
    y = model(x)
    _logger.info("Model output shape: %s", y.shape)

    # create pipeline
    pipeline = detectors.create_pipeline(
        args.pipeline,
        batch_size=args.batch_size,
        seed=args.seed,
        transform=test_transform,
        limit_fit=args.limit_fit,
        num_workers=3,
    )

    if "vit" in args.model and "pooling_op_name" in args.method_kwargs:
        args.method_kwargs["pooling_op_name"] = "getitem"

    # create detector
    method = detectors.create_detector(args.method, model=model, **args.method_kwargs)
    # run pipeline
    pipeline_results = pipeline.run(method)
    # print results
    print(pipeline.report(pipeline_results["results"]))

    if not args.debug:
        # save results to file
        results = {
            "model": args.model,
            "method": args.method,
            **pipeline_results["results"],
            "method_kwargs": args.method_kwargs,
        }
        filename = os.path.join(RESULTS_DIR, args.pipeline, "results.csv")
        detectors.utils.append_results_to_csv_file(results, filename)

        scores = pipeline_results["scores"]
        labels = pipeline_results["labels"]

        results = {
            "model": args.model,
            "in_dataset_name": pipeline.in_dataset_name,
            "out_datasets_names": pipeline.out_datasets_names,
            "method": args.method,
            "method_kwargs": args.method_kwargs,
            "scores": scores.numpy().tolist(),
            "labels": labels.numpy().tolist(),
        }
        filename = os.path.join(RESULTS_DIR, args.pipeline, "scores.csv")
        detectors.utils.append_results_to_csv_file(results, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="msp")
    parser.add_argument("--method_kwargs", type=json.loads, default={}, help='{"temperature":1000, "eps":0.00014}')
    parser.add_argument("--pipeline", type=str, default="ood_benchmark_cifar10")
    parser.add_argument("--model", type=str, default="resnet18_cifar10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--limit_fit", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--idx", type=int, default=None)
    args = parser.parse_args()

    methods = [
        "msp",
        "energy",
        "kl_matching",
        "igeood_logits",
        "max_logits",
        "react",
        "dice",
        "vim",
        "gradnorm",
        "mahalanobis",
        "relative_mahalanobis",
        "knn_euclides",
        "maxcosine",
        "odin",
        "doctor",
        "projection",
        "argm",
    ]
    if args.idx is not None:
        args.method = methods[args.idx]

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _logger.info(json.dumps(args.__dict__, indent=2))
    if args.method == "react_projection":
        args.method_kwargs["features_nodes"] = ["layer1", "layer2", "layer3", "clip", "fc"]
    if "vit" in args.model and "projection" in args.method:
        # args.method_kwargs["features_nodes"] = [f"blocks.{l}" for l in range(5, 12)] + ["fc_norm", "head"]
        args.method_kwargs["features_nodes"] = ["fc_norm", "head"]
    main(args)
