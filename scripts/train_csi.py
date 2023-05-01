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
from scripts.train_ssl import TwoCropTransform, knn_ssl_eval

_logger = logging.getLogger(__name__)


class CSIModel(nn.Module):
    def __init__(self, encoder, input_features_dim: int, output_features_dim: int = 128, **kwargs):
        super(CSIModel, self).__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(input_features_dim, input_features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_features_dim, output_features_dim),
        )
        self.shift_layer = nn.Linear(input_features_dim, 2)

    def forward(self, x):
        return F.normalize(self.projector(self.encoder.forward(x)), dim=-1), self.shift_layer(self.encoder.forward(x))


class Rotation(nn.Module):
    def __init__(self, max_range=4):
        super(Rotation, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None):
        if aug_index is None:
            aug_index = np.random.randint(4)

            output = torch.rot90(input, aug_index, (2, 3))

            _prob = input.new_full((input.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1 - _mask) * output

        else:
            aug_index = aug_index % self.max_range
            output = torch.rot90(input, aug_index, (2, 3))

        return output


def trainer_csi(csi_model, shift_trans, training_mode, dataloader, criterion, optimizer, lr_scheduler=None):
    csi_model.train()

    avg_loss = 0.0
    for i, data in enumerate(dataloader):
        images, target = data
        images1 = torch.cat([shift_trans(images[0], k) for k in range(4)])
        images2 = torch.cat([shift_trans(images[1], k) for k in range(4)])
        shift_labels = torch.cat([torch.ones_like(target) * k for k in range(4)], 0)  # B -> 4B
        shift_labels = shift_labels.repeat(2)
        images = torch.cat([images1, images2], dim=0)
        batch_size = target.shape[0] * 2
        _logger.debug("Images device: %s", images.device)

        features = csi_model(images)
        f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        _logger.debug("Features shape: %s", features.shape)
        if training_mode == "supcsi":
            loss = criterion.forward(features, shift_labels, target)
        elif training_mode == "csi":
            loss = criterion.forward(features, shift_labels)
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
    csi_model = CSIModel(model, model.num_features, args.output_features_dim)
    # get transform
    data_config = timm.data.resolve_data_config(model.default_cfg)
    test_transform = timm.data.create_transform(**data_config)
    data_config["is_training"] = True
    train_transform = timm.data.create_transform(**data_config, color_jitter=None)
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
    shift_transform = Rotation()

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
    optimizer = torch.optim.SGD(csi_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    warmup_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.01,
        max_lr=args.lr,
        step_size_up=args.warmup_epochs * len(train_dataloader),
    )

    # accelerate
    csi_model, optimizer, train_dataloader, val_dataloader, warmup_lr_scheduler = accelerator.prepare(
        csi_model, optimizer, train_dataloader, val_dataloader, warmup_lr_scheduler
    )

    # warmup iterations
    progress_bar = tqdm(
        range(args.warmup_epochs), desc="Warmup", disable=not accelerator.is_local_main_process, dynamic_ncols=True
    )
    for epoch in progress_bar:
        train_results = trainer_csi(
            csi_model, shift_transform, args.training_mode, train_dataloader, criterion, optimizer, warmup_lr_scheduler
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
    train_loss = 1e10
    epoch = 0
    progress_bar = tqdm(
        range(args.epochs), desc="Train", disable=not accelerator.is_local_main_process, dynamic_ncols=True
    )
    for epoch in progress_bar:
        train_results = trainer_csi(
            csi_model, shift_transform, args.training_mode, train_dataloader, criterion, optimizer, lr_scheduler
        )
        train_loss = train_results["loss"]

        if epoch % args.validation_frequency == 0:
            eval_results = knn_ssl_eval(model, val_dataloader, val_dataset_size)
            val_acc = eval_results["acc"]
            if val_acc > best_acc:
                best_acc = val_acc
                if accelerator.is_main_process:
                    _logger.debug("Saving best model with acc: %s", val_acc)
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

    parser.add_argument("--model", type=str, default="resnet34_csi_cifar10")

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
