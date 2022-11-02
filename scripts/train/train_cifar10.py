import argparse
import json
import logging
import os

import detectors
import torch
import torch.utils.data
from detectors.data import get_dataset
from detectors.data.cifar_wrapper import default_cifar10_test_transform
from detectors.trainer import trainer_classification
from detectors.trainer_utils import get_criterion_cls, get_optimizer_cls, get_scheduler_cls
from detectors.utils import str_to_dict


logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    folder_name = args.model if args.dataset in args.model else f"{args.model}_{args.dataset}"
    save_root = os.path.join(detectors.config.CHECKPOINTS_DIR, folder_name, str(args.seed))
    os.makedirs(save_root, exist_ok=True)

    train_transform = default_cifar10_test_transform()
    test_transform = default_cifar10_test_transform()

    train_dataset = get_dataset(args.dataset, split="train", download=True, transform=train_transform)
    val_dataset = get_dataset(args.dataset, split="test", download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = detectors.models.create_model(args.model, num_classes=10, weights=None)
    criterion = get_criterion_cls(args.criterion)(**args.criterion_kwargs)
    optimizer = get_optimizer_cls(args.optimizer)(model.parameters(), lr=args.lr, **args.optimizer_kwargs)  # type: ignore
    scheduler = get_scheduler_cls(args.scheduler)(optimizer, **args.scheduler_kwargs)  # type: ignore

    trainer_classification(
        model,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        val_loader,
        max_train_epochs=args.max_train_epochs,
        validation_frequency=args.validation_frequency,
        seed=args.seed,
        save_root=save_root,
    )

    # save hyper parameters
    with open(os.path.join(save_root, "hyperparameters.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_epochs", type=int, default=10)
    parser.add_argument("--validation_frequency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--criterion", type=str, default="CrossEntropyLoss")
    parser.add_argument("--criterion_kwargs", type=str_to_dict, default={})

    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer_kwargs", type=str_to_dict, default={})

    parser.add_argument("--scheduler", type=str, default="StepLR")
    parser.add_argument("--scheduler_kwargs", type=str_to_dict, default={"step_size": 1, "gamma": 0.7})

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
            args.__dict__.update(config)

    logging.basicConfig(
        format="---> %(levelname)s - %(name)s - %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    logger.info(json.dumps(args.__dict__, indent=2))
    main(args)
