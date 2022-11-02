from typing import Callable

import torch
from accelerate import Accelerator
from torch import Tensor, nn, optim


def get_criterion_cls(criterion_name: str) -> nn.modules.loss._Loss:
    return getattr(nn, criterion_name)


def get_optimizer_cls(optimizer_name: str) -> optim.Optimizer:
    return getattr(optim, optimizer_name)


def get_scheduler_cls(scheduler_name: str) -> optim.lr_scheduler._LRScheduler:
    return getattr(optim.lr_scheduler, scheduler_name)


def training_iteration(
    inputs: Tensor,
    targets: Tensor,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    criterion: Callable,
    accelerator: Accelerator,
):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return {"loss": loss.item()}


def validation_iteration(
    inputs: Tensor, targets: Tensor, model: nn.Module, criterion: Callable, accelerator: Accelerator
):
    with torch.no_grad():
        outputs = model(inputs)
    predictions = torch.argmax(outputs, dim=1)
    predictions, targets = accelerator.gather_for_metrics((predictions, targets))
    loss = criterion(outputs, targets)
    return {"loss": loss, "accuracy": (predictions == targets).float().sum()}


def save_model(model: nn.Module, accelerator: Accelerator, filename: str):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), filename)
