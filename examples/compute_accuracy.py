import argparse
import json
import logging
import os

import timm
import timm.data
import torch
import torch.utils.data
from tqdm import tqdm

import detectors
from detectors.config import RESULTS_DIR

_logger = logging.getLogger(__name__)


def topk_accuracy(preds, labels, k=5):
    topk = torch.topk(preds, k=k, dim=1)
    topk_preds = topk.indices
    topk_labels = labels.unsqueeze(1).expand_as(topk_preds)
    return (topk_preds == topk_labels).any(dim=1).float().mean().item()


def main(args):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        # try mps
        device = "mps"
    # create model
    model = timm.create_model(args.model, pretrained=True)
    model.to(device)
    print(model.default_cfg)
    data_config = timm.data.resolve_data_config(model.default_cfg)
    test_transform = timm.data.create_transform(**data_config)
    data_config["is_training"] = True
    train_transform = timm.data.create_transform(**data_config, color_jitter=None)

    _logger.info("Test transform: %s", test_transform)
    _logger.info("Train transform: %s", train_transform)

    dataset = detectors.create_dataset(args.dataset, split=args.split, transform=test_transform, download=True)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    x = x.to(device)
    with torch.no_grad():
        y = model(x)

    num_classes = y.shape[1]
    if args.dataset == "imagenet_r":
        mask = dataset.imagenet_r_mask
    else:
        mask = range(num_classes)

    all_preds = torch.empty((len(dataset), num_classes), dtype=torch.float32)
    all_labels = torch.empty(len(dataset), dtype=torch.long)
    _logger.info(f"Shapes: {all_preds.shape}, {all_labels.shape}")
    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        inputs, labels = batch
        inputs = inputs.to(device)
        # print(labels)
        with torch.no_grad():
            outputs = model(inputs)
            # print(outputs)
        outputs = torch.softmax(outputs, dim=1)
        all_preds[i * args.batch_size : (i + 1) * args.batch_size] = outputs.cpu()
        all_labels[i * args.batch_size : (i + 1) * args.batch_size] = labels.cpu()
        if args.debug:
            _logger.info("Labels: %s", labels)
            _logger.info("Predictions: %s", outputs.argmax(1))
            break

    top1 = topk_accuracy(all_preds[:, mask], all_labels, k=1) * 100
    top5 = topk_accuracy(all_preds[:, mask], all_labels, k=5) * 100
    _logger.info(torch.sum(torch.argmax(all_preds, dim=1) == all_labels) / len(all_labels))
    _logger.info(f"Top-1 accuracy: {top1:.4f}")
    _logger.info(f"Top-5 accuracy: {top5:.4f}")

    if not args.debug:
        # save results to file
        results = {
            "model": args.model,
            "dataset": args.dataset,
            "split": args.split,
            "top1_acc": top1,
            "top5_acc": top5,
        }
        filename = os.path.join(RESULTS_DIR, "accuracy", "results.csv")
        detectors.utils.append_results_to_csv_file(results, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50.tv_in1k")
    parser.add_argument("--dataset", type=str, default="imagenet1k")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    _logger.info(json.dumps(args.__dict__, indent=2))

    main(args)
