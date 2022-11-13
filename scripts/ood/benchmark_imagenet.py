import argparse
import torchvision
import json
import logging
import os
import time

import detectors
import torch
import torch.fx
from detectors.utils import str_to_dict
from torchvision.models.feature_extraction import get_graph_node_names


logger = logging.getLogger(__name__)


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    save_root = os.path.join(args.logging_dir, args.model, str(args.seed))
    os.makedirs(save_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = detectors.create_model(args.model, weights=True)
    if args.debug:
        logger.debug(model)
        logger.debug(get_graph_node_names(model)[0])
        # logger.debug(torch.fx.symbolic_trace(model).code)

    # TODO: fetch the good transform for the model

    model.to(device)
    model.eval()
    methods = {
        m: detectors.create_ood_detector(m, model, **(args.methods_kwargs[m] if m in args.methods_kwargs else {}))
        for m in args.methods
    }
    for m in methods.values():
        logger.info(m)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(args.resize_size),
            torchvision.transforms.CenterCrop(args.crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=args.mean, std=args.std),
        ]
    )

    start_time = time.time()
    pipeline = detectors.pipeline(
        "ood-imagenet",
        device=device,
        limit_fit=args.limit_fit,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    results = pipeline.benchmark(methods)
    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")

    # save results
    with open(os.path.join(save_root, "ood-imagenet.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--methods", nargs="+", default=["msp", "odin"])
    parser.add_argument(
        "--methods_kwargs", type=str_to_dict, default={}, help='{"odin": {"temperature":1000, "eps":0.00014}}'
    )

    parser.add_argument("--model", type=str, default="densenet121")
    parser.add_argument("--limit_fit", type=int, default=None)

    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--mean", nargs="+", type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", nargs="+", type=float, default=[0.229, 0.224, 0.225])

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--logging_dir", type=str, default="logs/imagenet")

    args = parser.parse_args()

    logging.basicConfig(
        format="---> %(levelname)s - %(name)s - %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )
    logger.info(json.dumps(args.__dict__, indent=2))

    main(args)

    # models:
    # densenet121
    # vit_b_16
    # bitsr101 = resnetv2_101x1_bitm, resnetv2_101x3_bitm
    # python scripts/ood/benchmark_imagenet.py --methods knn_euclides energy --model resnetv2_101x1_bitm --batch_size 256 --methods_kwargs '{"react": {"features_nodes":["head.global_pool.flatten"],"graph_nodes_names":["head_global_pool_flatten"], "p":0.9}, "dice": {"penultimate_node":"head.global_pool.flatten", "last_layer_name":"head.fc"}, "knn_euclides":{"features_nodes":["head.global_pool.flatten"]}}' --limit_fit 20000 --num_workers 3
