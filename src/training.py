import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

import torch
from torch.utils.data import Dataset

from .stratego_utils import START


def train(model, train_dataloader, num_epochs, optimizer, criterion,
          val_dataloader=None, eval_metrics=None, num_print_decimals=5, 
          device=None, callbacks=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    print(f"Using device '{device}'")
    model = model.to(device)

    metrics_per_epoch = {"train_metrics": {eval_metric: [] for eval_metric, func in eval_metrics}}
    if val_dataloader is not None:
        metrics_per_epoch["val_metrics"] = {eval_metric: [] for eval_metric, func in eval_metrics}

    for epoch in range(1, num_epochs + 1):

        train_single_epoch(model, train_dataloader, optimizer, criterion, device)

        message = f"Epoch: {epoch}"

        train_preds, y_train = predict_and_gather_labels(model, train_dataloader, device)

        for eval_metric, func in eval_metrics:
            score = func(y_train, train_preds)
            metrics_per_epoch["train_metrics"][eval_metric].append(score)
            message += f", Train {eval_metric}: {round(score, num_print_decimals)}"

        if val_dataloader is not None:

            val_preds, y_val = predict_and_gather_labels(model, val_dataloader, device)

            for eval_metric, func in eval_metrics:
                score = func(y_val, val_preds)
                metrics_per_epoch["val_metrics"][eval_metric].append(score)
                message += f", Val {eval_metric}: {round(score, num_print_decimals)}"

        print(message)

        if callbacks is None:
            continue
        if any(callback(metrics_per_epoch) for callback in callbacks):
            break

    return metrics_per_epoch


def train_single_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        if isinstance(output, tuple):
            output = output[0]
        loss = criterion(output, targets)
        loss.backward()
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
    model_info = {'model_state_dict': model.state_dict()}
    for key, value in to_save.items():
        if hasattr(value, 'state_dict'):
            model_info[f"{key}_state_dict"] = value.state_dict()
        else:
            model_info[key] = value
    torch.save(model_info, save_path)


def plot_metric_curves(train_metric_per_epoch, val_metric_per_epoch, best_epoch=None,
                       fig_size=(6, 4), title=None, metric_label=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    x_axis_values = list(range(1, len(train_metric_per_epoch) + 1))

    ax.plot(x_axis_values, train_metric_per_epoch, color="blue", label=f"Train Metric")
    ax.plot(x_axis_values, val_metric_per_epoch, color="red", label=f"Validation Metric")
    if best_epoch is not None:
        ax.axvline(x=best_epoch, linestyle="--", color="green", label="Best Epoch")
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel(metric_label if metric_label is not None else "Metric")
    ax.set_xlabel("Epochs")


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
        self.setups = [torch.tensor([START] + list(setup), dtype=torch.long) for setup in setups]

    def __getitem__(self, item):
        return self.setups[item]

    def __len__(self):
        return len(self.setups)


class SetupsDatasetWrapper(Dataset):

    mirror_indices = ([0] + list(range(10, 0, -1)) + list(range(20, 10, -1))
                          + list(range(30, 20, -1)) + list(range(40, 30, -1)))
            
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