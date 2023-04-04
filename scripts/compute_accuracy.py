import argparse
import json
import logging
import time

import accelerate
import numpy as np
import timm
import torch
import torch.utils.data
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

import detectors

_logger = logging.getLogger(__name__)


@torch.no_grad()
def main(args):
    accelerator = accelerate.Accelerator()

    model = timm.create_model(args.model, pretrained=True)
    data_config = resolve_data_config(model.default_cfg)
    transform = create_transform(**data_config)
    _logger.info(transform)

    model.eval()
    model = accelerator.prepare(model)

    dataset = detectors.create_dataset(args.dataset, split=args.split, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=accelerator.num_processes
    )
    dataloader = accelerator.prepare(dataloader)

    inference_time = []
    start_time = time.time()
    acc = 0
    progress_bar = tqdm(dataloader, desc="Inference", disable=not accelerator.is_local_main_process)
    for x, labels in dataloader:
        t1 = time.time()
        outputs = model(x)
        t2 = time.time()
        _, preds = torch.max(outputs, 1)

        preds, labels = accelerator.gather_for_metrics((preds, labels))
        acc += torch.sum((preds.cpu() == labels.cpu()).float()).item()

        inference_time.append(t2 - t1)
        progress_bar.update()
    progress_bar.close()
    accelerator.wait_for_everyone()

    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Accuracy: {acc / len(dataset)}")
    print(f"Average inference time: {np.mean(inference_time)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="densenet121")
    parser.add_argument("--dataset", type=str, default="imagenet1k")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    _logger.info(json.dumps(args.__dict__, indent=2))

    main(args)
