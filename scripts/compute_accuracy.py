import argparse
import json
import logging
import time

import accelerate
import numpy as np
import timm
import torch
import torch.utils.data
from sklearn.neighbors import KNeighborsClassifier
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

import detectors

_logger = logging.getLogger(__name__)


@torch.no_grad()
def main(args):
    if "supcon" in args.model or "simclr" in args.model:
        args.ssl = True
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
    all_outputs = []
    all_labels = []
    start_time = time.time()
    progress_bar = tqdm(dataloader, desc="Inference", disable=not accelerator.is_local_main_process)
    for x, labels in dataloader:
        t1 = time.time()
        outputs = model(x)
        t2 = time.time()

        outputs, labels = accelerator.gather_for_metrics((outputs, labels))
        all_outputs.append(outputs.cpu())
        all_labels.append(labels.cpu())
        inference_time.append(t2 - t1)
        progress_bar.update()

    progress_bar.close()
    accelerator.wait_for_everyone()

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if not args.ssl:
        _, preds = torch.max(all_outputs, 1)
    else:
        features = all_outputs.cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        estimator = KNeighborsClassifier(20, metric="cosine").fit(features, all_labels)
        preds = estimator.predict(features)
        preds = torch.from_numpy(preds)
        all_labels = torch.from_numpy(all_labels)

    acc = torch.mean((preds.cpu() == all_labels.cpu()).float()).item()

    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Accuracy: {acc}")
    print(f"Average inference time: {np.mean(inference_time)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="densenet121")
    parser.add_argument("--dataset", type=str, default="imagenet1k")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ssl", action="store_true")

    args = parser.parse_args()

    _logger.info(json.dumps(args.__dict__, indent=2))

    main(args)
