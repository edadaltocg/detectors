import argparse
import json
import logging
import os

import timm
import timm.data

import detectors
from detectors.utils import str_to_dict

logger = logging.getLogger(__name__)


def main(args):
    print(f"Running {args.pipeline} pipeline on {args.model} model")
    # create model
    model = timm.create_model(args.model, pretrained=True, num_classes=10)
    data_config = timm.data.resolve_data_config(model.default_cfg)
    test_transform = timm.data.create_transform(**data_config)
    # create pipeline
    pipeline = detectors.create_pipeline(
        args.pipeline, batch_size=args.batch_size, seed=args.seed, transform=test_transform, limit_fit=args.limit_fit
    )
    filename = os.path.join(args.save_root, f"{args.pipeline}.csv")
    for method_name in args.methods:
        print(f"Method: {method_name}")
        # create detector
        method = detectors.create_detector(method_name, model=model, **args.methods_kwargs.get(method_name, {}))
        # run pipeline
        results = pipeline.run(method)
        # print results
        print(pipeline.report(results))
        # save results to file
        results = {
            "model": args.model,
            "method": method_name,
            **results,
            "method_kwargs": args.methods_kwargs.get(method_name, {}),
        }
        # detectors.utils.append_results_to_csv_file(results, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, nargs="+", default=["msp", "random", "odin"])
    parser.add_argument(
        "--methods_kwargs", type=str_to_dict, default={}, help='{"odin": {"temperature":1000, "eps":0.00014}}'
    )
    parser.add_argument("--pipeline", type=str, default="ood_cifar10_benchmark")
    parser.add_argument("--model", type=str, default="resnet18_cifar10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--limit_fit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_root", type=str, default="results")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    logger.info(json.dumps(args.__dict__, indent=2))

    main(args)
