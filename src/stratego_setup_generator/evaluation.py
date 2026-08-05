import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import torch
from torch import nn
from torch.utils.data import TensorDataset

from .constants import (
    PIECE_TO_STR, NUM_SETUP_SQUARES, NUM_PIECE_TYPES, PIECE_COUNTS
)
from .plotting import create_plot_grid


def plot_per_square_distr_comparison(
        real_setups_df, generated_setups_df,
        kl_div=None, num_columns=10, width=5, height=3.5):

    fig, axes = create_plot_grid(NUM_SETUP_SQUARES, num_columns, width, height)
    axes = axes.ravel()

    for square, ax in zip(range(NUM_SETUP_SQUARES), axes):
        distr_real = (
            real_setups_df.iloc[:, square]
                    .value_counts(normalize=True)
                    .sort_index()
                    .rename(index=PIECE_TO_STR)
        )
        distr_generated = (
            generated_setups_df.iloc[:, square]
                    .value_counts(normalize=True)
                    .sort_index()
                    .rename(index=PIECE_TO_STR)
        )
        df = pd.DataFrame(
            {'real setups': distr_real, 'generated setups': distr_generated}
        )
        df.plot.bar(ax=ax, rot=0)
        title = f"Square {square}"
        if kl_div is not None:
            title += f" (KL={round(kl_div.loc[square], 4)})"
        ax.set_title(title)
        ax.set_xlabel('')

    for i in range(NUM_SETUP_SQUARES, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def kl_divergence(p, q, smoothing=1e-12):
    p = p + smoothing
    q = q + smoothing
    p /= p.sum()
    q /= q.sum()
    return (p * np.log(p / q)).sum()


def compute_kl_div_single_squares(real_setups_df, generated_setups_df):

    kl_divergences = []

    for square in range(NUM_SETUP_SQUARES):
        p = (
            real_setups_df.iloc[:, square]
             .value_counts(normalize=True)
             .sort_index()
             .rename(index=PIECE_TO_STR)
             .to_numpy()
        )
        q = (
            generated_setups_df.iloc[:, square]
             .value_counts(normalize=True)
             .sort_index()
             .rename(index=PIECE_TO_STR)
             .to_numpy()
        )
        kl_divergences.append(kl_divergence(p, q))

    index = pd.Index(range(NUM_SETUP_SQUARES), name="sq")
    return pd.Series(kl_divergences, name="kl_div", index=index)


def compute_kl_div_square_pairs(real_setups_df, generated_setups_df):

    valid_piece_pairs = [
        (PIECE_TO_STR[piece1], PIECE_TO_STR[piece2])
        for piece1 in range(NUM_PIECE_TYPES)
        for piece2 in range(NUM_PIECE_TYPES)
        if piece1 != piece2 or PIECE_COUNTS[piece1] >= 2
    ]
    square_pairs = list(itertools.combinations(range(NUM_SETUP_SQUARES), 2))
    kl_divergences = []

    for sq1, sq2 in square_pairs:
        p = (
            real_setups_df.iloc[:, [sq1, sq2]]
             .replace(PIECE_TO_STR)
             .value_counts(normalize=True)
             .reindex(valid_piece_pairs, fill_value=0)
             .to_numpy()
        )
        q = (
            generated_setups_df.iloc[:, [sq1, sq2]]
             .replace(PIECE_TO_STR)
             .value_counts(normalize=True)
             .reindex(valid_piece_pairs, fill_value=0)
             .to_numpy()
        )
        kl_divergences.append(kl_divergence(p, q))

    index = pd.MultiIndex.from_tuples(square_pairs, names=("sq1", "sq2"))
    return pd.Series(kl_divergences, name="kl_div", index=index)


def find_most_overlapping(from_setups, to_setups, batch_size=256):

    most_overlapping = np.empty(len(from_setups), dtype='int')
    num_overlaps = np.empty(len(from_setups), dtype='int')

    for start in range(0, len(from_setups), batch_size):
        end = start + batch_size
        batch = from_setups[start:end]
        num_overlaps = np.sum(
            batch[:, None, :] == to_setups[None, :, :],
            axis=2,
            dtype=np.uint8,
        )
        most_overlapping[start:end] = np.argmax(num_overlaps, axis=1)
        num_overlaps[start:end] = (
            num_overlaps[np.arange(len(batch)), most_overlapping[start:end]]
        )

    return most_overlapping, num_overlaps


class LSTMClassifier(nn.Module):

    def __init__(
        self, hidden_size, embedding_dim, num_layers=1, bidirectional=True):

        super().__init__()
        self.embedding = nn.Embedding(NUM_PIECE_TYPES, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        fc_out_input_size = 2 * hidden_size if bidirectional else hidden_size
        self.fc_out = nn.Linear(fc_out_input_size, 1)

    def forward(self, x, hidden_state=None):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x, hidden_state = self.lstm(x, hidden_state)
        if self.lstm.bidirectional:
            hidden_size = self.lstm.hidden_size
            x_forward_last = x[:, -1, :hidden_size]
            x_backward_first = x[:, 0, hidden_size:]
            x = torch.cat([x_forward_last, x_backward_first], dim=-1)
        else:
            x = x[:, -1, :]
        x = self.fc_out(x)
        return x, hidden_state


def binary_log_loss_from_logits(y_true, y_pred):
    return log_loss(y_true, 1 / (1 + np.exp(-y_pred)))


def create_classification_dataset(X, y):
    X = torch.tensor(X.to_numpy(), dtype=torch.long)
    y = torch.tensor(y.to_numpy(), dtype=torch.float32)
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    return TensorDataset(X, y)