import argparse
import json
import logging
import os
import time

from tqdm import tqdm

import detectors
import torch
import torch.fx
import torch.utils.data
import torchvision
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

logger = logging.getLogger(__name__)


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = detectors.create_model(args.model, weights="DEFAULT")
    model.to(device)
    model.eval()
    try:
        config = resolve_data_config({}, model=model)
        config["is_training"] = False
        transform = create_transform(**config)
    except:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(args.resize_size),
                torchvision.transforms.CenterCrop(args.crop_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    logger.info(transform)
    dataset = detectors.get_dataset("imagenet1k", split="val", transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    start_time = time.time()
    acc = 0
    for x, labels in tqdm(dataloader):
        x = x.to(device)
        outputs = model(x)
        _, preds = torch.max(outputs, 1)
        acc += torch.sum(preds.cpu() == labels.data).item()
    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")
    logger.info(f"Accuracy: {acc / len(dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="densenet121")

    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    logging.basicConfig(format="---> %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    logger.info(json.dumps(args.__dict__, indent=2))

    main(args)
