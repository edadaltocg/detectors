import json
import logging
import os
from typing import Callable, Tuple

import accelerate
import torch
import torch.utils.data
from accelerate import Accelerator
from torch import Tensor, nn, optim
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def get_criterion_cls(criterion_name: str) -> nn.modules.loss._Loss:
    return getattr(nn, criterion_name)


def get_optimizer_cls(optimizer_name: str) -> optim.Optimizer:
    return getattr(optim, optimizer_name)


def get_scheduler_cls(scheduler_name: str) -> optim.lr_scheduler._LRScheduler:
    return getattr(optim.lr_scheduler, scheduler_name)


def training_iteration(
    batch: Tuple[Tensor, Tensor],
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: Callable,
    accelerator: Accelerator,
):
    inputs, targets = batch

    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    accelerator.backward(loss)
    optimizer.step()
    return {"loss": loss.item()}


def validation_iteration(batch: Tuple[Tensor, Tensor], model: nn.Module, criterion: Callable, accelerator: Accelerator):
    inputs, targets = batch
    with torch.no_grad():
        outputs = model(inputs)
    predictions = torch.argmax(outputs, dim=1)
    predictions, targets = accelerator.gather_for_metrics((predictions, targets))
    loss = criterion(outputs, targets).item()
    acc = (predictions == targets).float().sum().item()
    return {"loss": loss, "accuracy": acc}


def save_model(model: nn.Module, accelerator: Accelerator, filename: str):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), filename)


def trainer_classification(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: Callable,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    save_root: str,
    training_function=training_iteration,
    validation_function=validation_iteration,
    epochs=10,
    validation_frequency=1,
    seed: int = 42,
):
    os.makedirs(save_root, exist_ok=True)
    _logger.info(f"Saving model and progress to {save_root}")

    accelerate.utils.set_seed(seed)
    accelerator = Accelerator(
        log_with=["all"], logging_dir=os.path.join(save_root, "logs"), step_scheduler_with_optimizer=False
    )
    accelerator.init_trackers("")
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_accuracy = 0.0
    step = 0
    progress_bar = tqdm(range(epochs), disable=not accelerator.is_local_main_process, dynamic_ncols=True)
    for epoch in progress_bar:
        # train
        avg_loss = 0.0
        model.train()
        for batch in train_loader:
            step += 1
            tr_obj = training_function(batch, model, optimizer, criterion, accelerator)
            tr_loss = tr_obj["loss"]
            lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_description_str(f"{step}it, loss={tr_loss:.4f}, lr={lr:.4f}")
            accelerator.log({f"train/{k}": v for k, v in tr_obj.items()}, step=step)
            accelerator.log({"lr": lr}, step=step)
            avg_loss += tr_loss
        avg_loss /= len(train_loader)

        # validate
        if (epoch + 1) % validation_frequency == 0 or epoch == 0 or epoch == epochs - 1:
            model.eval()
            val_acc = 0
            val_loss = 0
            total = 0
            val_obj = {}
            for batch in val_loader:
                val_obj = validation_function(batch, model, criterion, accelerator)

                val_acc += val_obj["accuracy"]
                val_loss += val_obj["loss"]
                total += len(batch[0])  # temporary fix

            val_acc /= total
            val_loss /= total

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                save_model(model, accelerator, os.path.join(save_root, "best.pth"))

            progress_bar.set_postfix({"val/loss": val_loss, "val/acc": val_acc, "best/acc": best_accuracy})
            accelerator.log({"val/acc": val_acc, "val/loss": val_loss}, step=epoch)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_acc)

        # sync accelerator after epoch
        accelerator.wait_for_everyone()
        progress_bar.update(1)

        if scheduler is not None:
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch)

    accelerator.end_training()
    # save model
    save_model(model, accelerator, os.path.join(save_root, "last.pth"))

    # save train results
    train_results = {"epoch": epoch, "train_loss": tr_loss}
    with open(os.path.join(save_root, "train_results.json"), "w") as f:
        json.dump(train_results, f)

    # save eval results
    eval_results = {"epoch": epoch, "best_accuracy": best_accuracy, "last_accuracy": val_acc, "eval_loss": val_loss}
    with open(os.path.join(save_root, "eval_results.json"), "w") as f:
        json.dump(eval_results, f)

    _logger.info(f"Training finished. Best accuracy: {best_accuracy:.4f}")
    _logger.info(f"Last model saved to {save_root}/last.pth")
    _logger.info(f"Best model saved to {save_root}/best.pth")
    _logger.info(f"Training logs saved to {save_root}/logs")
    _logger.info(f"Training results saved to {save_root}/train_results.json")
    _logger.info(f"Evaluation results saved to {save_root}/eval_results.json")
