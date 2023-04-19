import argparse
import json
import logging
import os
import sys

import accelerate
import numpy as np
import timm
import timm.data
import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils.data
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torchvision import transforms
from tqdm import tqdm

import detectors
import detectors.utils
from detectors.criterions import SupConLoss
from detectors.data.constants import (
    CIFAR10_DEFAULT_MEAN,
    CIFAR10_DEFAULT_STD,
    CIFAR100_DEFAULT_MEAN,
    CIFAR100_DEFAULT_STD,
)

_logger = logging.getLogger(__name__)


class SSLModel(nn.Module):
    def __init__(self, encoder, input_features_dim: int, output_features_dim: int = 128, **kwargs):
        super(SSLModel, self).__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(input_features_dim, input_features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_features_dim, output_features_dim),
        )

    def forward(self, x):
        return F.normalize(self.projector(self.encoder.forward(x)), dim=-1)


class TwoCropTransform:
    """Create two crops of the same image.

    References:
        https://github.com/HobbitLong/SupContrast
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

    def __repr__(self):
        return f"{self.__class__.__name__}(transform={self.transform.__repr__()})"


def knn_ssl_eval(model, val_loader, size_limit=None):
    """
    Evaluating knn accuracy in feature space.
    """
    model.eval()

    features = []
    labels = []
    for i, data in enumerate(val_loader):
        images, target = data
        # compute output
        with torch.no_grad():
            output = F.normalize(model(images), dim=-1)
        output = detectors.utils.sync_tensor_across_gpus(output)
        target = detectors.utils.sync_tensor_across_gpus(target)
        features.append(output.cpu())
        labels.append(target.cpu())

    features = torch.cat(features).numpy()[:size_limit]
    labels = torch.cat(labels).numpy()[:size_limit]
    # remove extra samples
    features = features[: len(labels)]
    cls = KNeighborsClassifier(20, metric="cosine").fit(features, labels)
    acc = np.mean(cross_val_score(cls, features, labels))

    return {"acc": acc}


def trainer_ssl(ssl_model, training_mode, dataloader, criterion, optimizer, lr_scheduler=None):
    ssl_model.train()

    avg_loss = 0.0
    for i, data in enumerate(dataloader):
        images, target = data
        images = torch.cat([images[0], images[1]], dim=0)
        batch_size = target.shape[0]
        _logger.debug("Images device: %s", images.device)

        features = ssl_model(images)
        f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        _logger.debug("Features shape: %s", features.shape)
        if training_mode == "supcon":
            # Supervised contrastive learning
            loss = criterion.forward(features, target)
        elif training_mode == "simclr":
            # SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
            loss = criterion.forward(features)
        else:
            raise ValueError("training mode not supported")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        avg_loss += loss.item()

        _logger.debug("Batch index: %s/%s", i, len(dataloader))
        _logger.debug("Batch loss: %s", loss.item())
        _logger.debug("Batch avg loss: %s", avg_loss / (i + 1))

    avg_loss /= len(dataloader)
    return {"loss": avg_loss}


def main(args):
    args.dataset = args.model.split("_")[-1]
    args.training_mode = args.model.split("_")[1]
    if args.debug:
        args.warmup_epochs = 1
        args.epochs = 1

    # save destination
    folder_name = args.model + f"_{args.seed}"
    save_root = os.path.join(detectors.config.CHECKPOINTS_DIR, folder_name)
    os.makedirs(save_root, exist_ok=True)

    accelerate.utils.set_seed(args.seed)
    accelerator = accelerate.Accelerator()
    accelerator.init_trackers(f"train_ssl_{args.model}", config=args.__dict__)

    if args.dataset == "cifar10":
        mean = CIFAR10_DEFAULT_MEAN
        std = CIFAR10_DEFAULT_STD
    elif args.dataset == "cifar100":
        mean = CIFAR100_DEFAULT_MEAN
        std = CIFAR100_DEFAULT_STD
    else:
        raise ValueError("dataset not supported")

    # model
    model = timm.create_model(args.model, pretrained=False, num_classes=0)
    ssl_model = SSLModel(model, model.num_features, args.output_features_dim)
    # get transform
    data_config = timm.data.resolve_data_config(model.default_cfg)
    test_transform = timm.data.create_transform(**data_config)
    data_config["is_training"] = True
    train_transform = timm.data.create_transform(**data_config, color_jitter=None)
    # TODO: implement this to timm's transforms
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_transform = TwoCropTransform(train_transform)

    _logger.info(f"train_transform: {train_transform}")
    _logger.info(f"test_transform: {test_transform}")

    # datasets
    train_dataset = detectors.create_dataset(args.dataset, split="train", download=True, transform=train_transform)
    val_dataset = detectors.create_dataset(args.dataset, split="test", download=True, transform=test_transform)
    val_dataset_size = len(val_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # criterion, optimizer, scheduler
    criterion = SupConLoss(temperature=0.5)
    optimizer = torch.optim.SGD(ssl_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    warmup_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.01,
        max_lr=args.lr,
        step_size_up=args.warmup_epochs * len(train_dataloader),
    )

    # accelerate
    ssl_model, optimizer, train_dataloader, val_dataloader, warmup_lr_scheduler = accelerator.prepare(
        ssl_model, optimizer, train_dataloader, val_dataloader, warmup_lr_scheduler
    )

    # warmup iterations
    progress_bar = tqdm(
        range(args.warmup_epochs), desc="Warmup", disable=not accelerator.is_local_main_process, dynamic_ncols=True
    )
    for epoch in progress_bar:
        train_results = trainer_ssl(
            ssl_model, args.training_mode, train_dataloader, criterion, optimizer, warmup_lr_scheduler
        )
        train_loss = train_results["loss"]
        progress_bar.set_postfix({"warmup/loss": train_loss})
        progress_bar.update()
    progress_bar.close()

    for p in optimizer.param_groups:
        p["lr"] = args.lr
        p["initial_lr"] = args.lr

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_dataloader), 1e-4)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # training iterations
    val_acc = 0
    best_acc = 0
    progress_bar = tqdm(
        range(args.epochs), desc="Train", disable=not accelerator.is_local_main_process, dynamic_ncols=True
    )
    for epoch in progress_bar:
        train_results = trainer_ssl(ssl_model, args.training_mode, train_dataloader, criterion, optimizer, lr_scheduler)
        train_loss = train_results["loss"]

        if epoch % args.validation_frequency == 0:
            eval_results = knn_ssl_eval(model, val_dataloader, val_dataset_size)
            val_acc = eval_results["acc"]
            if val_acc > best_acc and accelerator.is_main_process:
                _logger.debug("Saving best model with acc: %s", val_acc)
                best_acc = val_acc
                filename = os.path.join(save_root, "best.pth")
                # accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), filename)
                _logger.debug("Saved best model on %s", filename)

        # accelerator.log({"val/acc": val_acc, "train/loss": train_loss}, step=epoch)
        progress_bar.set_postfix({"train/loss": train_loss, "val/acc": val_acc, "best/acc": best_acc})
        progress_bar.update()
    progress_bar.close()

    if accelerator.is_main_process:
        # save last model
        filename = os.path.join(save_root, "last.pth")
        model = accelerator.unwrap_model(model)
        accelerator.save(model.state_dict(), filename)
        _logger.info("Saved last model on %s", filename)

        # save hyper parameters
        with open(os.path.join(save_root, "hyperparameters.json"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

        # save train results
        train_results = {"epoch": epoch, "train_loss": train_loss}
        with open(os.path.join(save_root, "train_results.json"), "w") as f:
            json.dump(train_results, f)

        # save eval results
        eval_results = {"epoch": epoch, "best_accuracy": best_acc, "last_accuracy": val_acc}
        with open(os.path.join(save_root, "eval_results.json"), "w") as f:
            json.dump(eval_results, f)

        _logger.info(f"Training finished. Best accuracy: {best_acc:.4f}")
        _logger.info(f"Last model saved to {save_root}/last.pth")
        _logger.info(f"Best model saved to {save_root}/best.pth")
        _logger.info(f"Training logs saved to {save_root}/logs")
        _logger.info(f"Training results saved to {save_root}/train_results.json")
        _logger.info(f"Evaluation results saved to {save_root}/eval_results.json")

    accelerator.end_training()
    _logger.info("Done! for rank %s", accelerator.process_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--model", type=str, default="resnet34_simclr_cifar10")
    parser.add_argument("--training_mode", type=str, default="simclr", choices=["simclr", "supcon"])

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=501)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--validation_frequency", type=int, default=50)

    parser.add_argument("--output_features_dim", type=int, default=128)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
            args.__dict__.update(config)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    _logger.info(json.dumps(args.__dict__, indent=2))
    main(args)
    sys.exit(0)
