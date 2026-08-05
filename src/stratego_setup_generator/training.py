import random

import numpy as np
from sklearn.metrics import log_loss

import torch
from torch.utils.data import Dataset

from .constants import START


def train(
    model,
    train_dataloader,
    num_epochs,
    optimizer,
    criterion,
    val_dataloaders=None,
    val_metrics=None,
    num_print_decimals=5,
    max_grad_norm=None,
    device=None,
    callbacks=None,
    verbose=True
    ):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if verbose:
        print(f"Using device '{device}'")

    model = model.to(device)

    val_dataloaders = val_dataloaders or []
    val_metrics = val_metrics or []
    callbacks = callbacks or []
    max_grad_norm = max_grad_norm or float('inf')

    metrics_per_epoch = {}
    for val_set_name, val_dataloader in val_dataloaders:
        metrics_per_epoch[val_set_name] = {
            val_metric: [] for val_metric, func in val_metrics
        }

    for epoch in range(1, num_epochs + 1):

        train_single_epoch(
            model, train_dataloader, optimizer,
            criterion, max_grad_norm, device
        )

        message_parts = []

        for val_set_name, val_dataloader in val_dataloaders:

            val_preds, y_val = (
                predict_and_gather_labels(model, val_dataloader, device)
            )
            for val_metric, func in val_metrics:
                score = func(y_val, val_preds)
                metrics_per_epoch[val_set_name][val_metric].append(float(score))

            formatted_metrics = ", ".join(
                f"{metric_name}={values[-1]:.{num_print_decimals}f}"
                for metric_name, values in metrics_per_epoch[val_set_name].items()
            )
            message_parts.append(f"{val_set_name}: {formatted_metrics}")

        message = f"[Epoch {epoch}] {' | '.join(message_parts)}"
        if verbose:
            print(message)

        if any(callback(metrics_per_epoch) for callback in callbacks):
            break

    return metrics_per_epoch


def train_single_epoch(
        model, dataloader, optimizer, criterion, max_grad_norm, device):

    model.train()

    for inputs, targets in dataloader:

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        output = model(inputs)
        if isinstance(output, tuple):
            output = output[0]

        loss = criterion(output, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()


@torch.no_grad()
def predict_and_gather_labels(model, dataloader, device):

    model.eval()

    predictions = []
    labels = []

    for inputs, targets in dataloader:

        inputs = inputs.to(device)

        output = model(inputs)
        if isinstance(output, tuple):
            output = output[0]

        predictions.append(output)
        labels.append(targets)

    predictions = torch.cat(predictions).squeeze(-1).cpu().numpy()
    labels = torch.cat(labels).squeeze(-1).cpu().numpy()

    return predictions, labels


def save_model(save_path, model, **to_save):
    if not save_path.endswith(".pth"):
        save_path += ".pth"
    checkpoint = {'model_state_dict': model.state_dict()}
    for key, value in to_save.items():
        if hasattr(value, 'state_dict'):
            checkpoint[f"{key}_state_dict"] = value.state_dict()
        else:
            checkpoint[key] = value
    torch.save(checkpoint, save_path)


class LearningRateCallback:

  def __init__(self, lr_scheduler, verbose=True):
      self.lr_scheduler = lr_scheduler
      self.verbose = verbose

  def __call__(self, metrics_per_epoch):
      self.lr_scheduler.step()
      if self.verbose:
          print(f"Learning Rate: {self.lr_scheduler.get_last_lr()}")
      return False


def log_loss_seq(y_true, y_pred):
    num_classes = y_pred.shape[1]
    y_true = y_true.reshape(-1)
    y_pred = y_pred.transpose(0, 2, 1).reshape(-1, num_classes)
    y_pred = np.exp(y_pred - np.max(y_pred, axis=-1, keepdims=True))
    y_pred /= np.sum(y_pred, axis=-1, keepdims=True)
    return log_loss(y_true, y_pred)


class SetupsDataset(Dataset):

    def __init__(self, setups):
        start_token_column = np.full((len(setups), 1), START)
        self.setups = torch.tensor(
            np.hstack([start_token_column, setups]), dtype=torch.long
        )

    def __getitem__(self, item):
        return self.setups[item]

    def __len__(self):
        return len(self.setups)


class SetupsDatasetWrapper(Dataset):

    mirror_indices = [
        0,
        *range(10, 0, -1),
        *range(20, 10, -1),
        *range(30, 20, -1),
        *range(40, 30, -1),
    ]
            
    def __init__(self, dataset, mirror_prob=0.0, random_state=None):
        self.dataset = dataset
        self.mirror_prob = mirror_prob if 0.0 <= mirror_prob <= 1.0 else -1.0
        self.rng = random.Random(random_state)

    def __getitem__(self, item):
        setup = self.dataset[item]
        if self.mirror_prob > 0.0 and self.rng.random() < self.mirror_prob:
            setup = setup[self.mirror_indices]
        return setup[:-1], setup[1:]

    def __len__(self):
        return len(self.dataset)