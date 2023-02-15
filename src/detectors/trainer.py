import os
from typing import Callable, Tuple

import accelerate
import torch
import torch.utils.data
from accelerate import Accelerator
from torch import Tensor, nn, optim
from tqdm import tqdm


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
    loss = criterion(outputs, targets)
    acc = (predictions == targets).float().sum()
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
    accelerate.utils.set_seed(seed)
    accelerator = Accelerator(log_with=["all"], logging_dir=os.path.join(save_root, "logs"))

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_accuracy = 0.0
    progress_bar = tqdm(range(epochs), total=epochs, disable=not accelerator.is_local_main_process, colour="yellow")
    for epoch in progress_bar:
        # train
        model.train()
        for batch in train_loader:
            tr_obj = training_function(batch, model, optimizer, criterion, accelerator)
            progress_bar.update(1)
            lr = scheduler.get_last_lr()
            progress_bar.set_description_str(f"Epoch: {epoch+1}/{epochs}, loss={tr_obj['loss']:.4f}, lr={lr[0]:.4f}")

        if scheduler is not None:
            scheduler.step()

        # validate
        if (epoch + 1) % validation_frequency == 0 or epoch == 0 or epoch == epochs - 1:
            model.eval()
            accuracy = 0
            loss = 0
            total = 0
            for batch in val_loader:
                val_obj = validation_function(batch, model, criterion, accelerator)

                accuracy += val_obj["accuracy"].item()
                loss += val_obj["loss"].item()
                total += len(batch[0])  # temporary fix

            accuracy /= total
            loss /= total

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_model(model, accelerator, os.path.join(save_root, "best.pth"))

            progress_bar.set_postfix({"val/acc": accuracy, "best/acc": best_accuracy})

    # save model
    save_model(model, accelerator, os.path.join(save_root, "last.pth"))
