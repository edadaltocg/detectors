import argparse
import json
import logging
import os

import timm
import timm.data
import torch
import torch.utils.data

import detectors
from detectors.data import create_dataset
from detectors.trainer import get_criterion_cls, get_optimizer_cls, get_scheduler_cls, trainer_classification
from detectors.utils import str_to_dict

_logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    # save destination
    folder_name = args.model + f"_{args.seed}"
    save_root = os.path.join(detectors.config.CHECKPOINTS_DIR, folder_name)

    # model
    model = timm.create_model(args.model, pretrained=args.pretrained)
    # get transform
    data_config = timm.data.resolve_data_config(model.default_cfg)
    test_transform = timm.data.create_transform(**data_config)
    data_config["is_training"] = True
    train_transform = timm.data.create_transform(**data_config, color_jitter=None)

    _logger.info(f"train_transform: {train_transform}")
    _logger.info(f"test_transform: {test_transform}")

    # datasets
    train_dataset = create_dataset(args.dataset, split="train", download=True, transform=train_transform)
    val_dataset = create_dataset(args.dataset, split="test", download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # criterion, optimizer, scheduler
    criterion = get_criterion_cls(args.criterion)(**args.criterion_kwargs)
    optimizer = get_optimizer_cls(args.optimizer)(model.parameters(), lr=args.lr, **args.optimizer_kwargs)
    scheduler = get_scheduler_cls(args.scheduler)(optimizer, **args.scheduler_kwargs)

    # train
    trainer_classification(
        model,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        val_loader,
        save_root=save_root,
        epochs=args.epochs,
        validation_frequency=args.validation_frequency,
        seed=args.seed,
    )

    # load best model
    model.load_state_dict(torch.load(os.path.join(save_root, "best.pth")))

    # save hyper parameters
    with open(os.path.join(save_root, "hyperparameters.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--model", type=str, default="resnet18_cifar10")
    parser.add_argument("--dataset", type=str, default="cifar10")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validation_frequency", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--criterion", type=str, default="CrossEntropyLoss")
    parser.add_argument("--criterion_kwargs", type=str_to_dict, default={})

    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer_kwargs", type=str_to_dict, default={})

    parser.add_argument("--scheduler", type=str, default="StepLR")
    parser.add_argument("--scheduler_kwargs", type=str_to_dict, default={"step_size": 30, "gamma": 0.1})

    parser.add_argument("--pretrained", action="store_true")

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
    model = main(args)
