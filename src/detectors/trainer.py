import os
from typing import Callable, Optional

import accelerate
import torch
import torch.utils.data
from accelerate import Accelerator
from detectors.trainer_utils import save_model, training_iteration, validation_iteration
from tqdm.auto import tqdm


def trainer_classification(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: Callable,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    max_train_epochs=10,
    validation_frequency=1,
    seed: int = 42,
    training_function=training_iteration,
    validation_function=validation_iteration,
    save_root: Optional[str] = None,
):
    accelerate.utils.set_seed(seed)
    accelerator = Accelerator()

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    val_loader = accelerator.prepare(val_loader)

    best_accuracy = 0.0
    progress_bar = tqdm(
        range(max_train_epochs), total=max_train_epochs, disable=not accelerator.is_local_main_process, colour="yellow"
    )
    for epoch in progress_bar:
        # train
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            tr_obj = training_function(inputs, targets, model, optimizer, scheduler, criterion, accelerator)
            progress_bar.update(1)
            progress_bar.set_description_str(f"Epoch: {epoch+1}/{max_train_epochs}, Loss: {tr_obj['loss']:.4f}")

        # validate
        if epoch % validation_frequency == 0:
            model.eval()
            accuracy = 0
            loss = 0
            total = 0
            for batch in val_loader:
                inputs, targets = batch
                val_obj = validation_function(inputs, targets, model, criterion, accelerator)

                accuracy += val_obj["accuracy"].item()
                loss += val_obj["loss"].item()
                total += len(inputs)

            accuracy /= total
            loss /= total

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_model(model, accelerator, os.path.join(save_root or "", "best.pth"))

            progress_bar.set_postfix({"val/loss": loss, "val/acc": accuracy, "best/acc": best_accuracy})

    # save model
    save_model(model, accelerator, os.path.join(save_root or "", "last.pth"))
