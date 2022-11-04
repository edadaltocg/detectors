import json
import pandas as pd
import os
import detectors
import argparse
import logging
import torch

from detectors.utils import str_to_dict

logger = logging.getLogger(__name__)


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    save_root = os.path.join(args.logging_dir, args.model, str(args.seed))
    os.makedirs(save_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = detectors.create_model(args.model, weights=None, device=device)
    model.to(device)
    methods = {
        m: detectors.create_ood_detector(m, model, **(args.methods_kwargs[m] if m in args.methods_kwargs else {}))
        for m in args.methods
    }
    pipeline = detectors.pipeline(
        "ood-cifar10", device=device, batch_size=args.batch_size, num_workers=args.num_workers
    )
    results = pipeline.benchmark(methods)

    # save results
    with open(os.path.join(save_root, "ood-cifar10.json"), "w") as f:
        json.dump(results, f)

    ## json to csv
    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.join(save_root, "ood-cifar10.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--methods", nargs="+", default=["msp", "odin"])
    parser.add_argument("--methods_kwargs", type=str_to_dict, default={})

    parser.add_argument("--model", type=str, default="resnet18_cifar10")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--logging_dir", type=str, default="logs/")

    args = parser.parse_args()

    logging.basicConfig(
        format="---> %(levelname)s - %(name)s - %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )
    logger.info(json.dumps(args.__dict__, indent=2))

    main(args)
