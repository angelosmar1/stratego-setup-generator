import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import torch
from torch import nn
from torch.utils.data import TensorDataset

from .stratego_utils import PIECE_TO_STR, NUM_SETUP_SQUARES, NUM_PIECE_TYPES
from .plotting import create_plot_grid


def plot_per_square_distributions(real_setups, generated_setups, num_columns=10, width=5, height=3.5):

    fig, ax = create_plot_grid(NUM_SETUP_SQUARES, num_columns, width, height)

    for square in range(NUM_SETUP_SQUARES):
        row, column = divmod(square, num_columns)
        distr1 = (real_setups.iloc[:, square].value_counts(normalize=True)
                  .sort_index().rename(index=PIECE_TO_STR))
        distr2 = (generated_setups.iloc[:, square].value_counts(normalize=True)
                  .sort_index().rename(index=PIECE_TO_STR))
        df = pd.DataFrame({'real setups': distr1, 'generated setups': distr2})
        df.plot.bar(ax=ax[row][column], rot=0)
        ax[row][column].set_title(f"Square {square}")
        ax[row][column].set_xlabel('')

    for i in range(NUM_SETUP_SQUARES, len(fig.axes)):
        row, column = divmod(i, num_columns)
        ax[row][column].axis('off')

    plt.tight_layout()
    plt.show()
    

def compute_nearest_neighbors(from_setups, to_setups, batch_size=500):

    nearest_neighbors = np.zeros(shape=(len(from_setups),), dtype='int')
    max_overlaps = np.zeros(shape=(len(from_setups),), dtype='int')

    for start in range(0, len(from_setups), batch_size):
        end = start + batch_size
        batch = from_setups[start:end]
        num_overlaps = np.sum(batch[:, np.newaxis, :] == to_setups[np.newaxis, :, :], axis=2)
        nearest_neighbors[start:end] = np.argmax(num_overlaps, axis=1)
        max_overlaps[start:end] = num_overlaps[np.arange(len(batch)), nearest_neighbors[start:end]]

    return nearest_neighbors, max_overlaps


class LSTMClassifier(nn.Module):

    def __init__(self, hidden_size, embedding_dim, num_layers=1, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(NUM_PIECE_TYPES, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc_out = nn.Linear(2 * hidden_size, 1)
        else:
            self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        out, hidden = self.lstm(x, hidden)
        if self.lstm.bidirectional:
            out = torch.cat([out[:, -1, :out.shape[-1]//2], out[:, 0, out.shape[-1]//2:]], dim=-1)
        else:
            out = out[:, -1, :]
        out = self.fc_out(out)
        return out, hidden


def log_loss_from_logits(y_true, y_pred):
    return log_loss(y_true, 1 / (1 + np.exp(-y_pred)))


def create_classification_dataset(X, y):
    X = torch.tensor(X.to_numpy(), dtype=torch.long)
    y = torch.tensor(y.to_numpy(), dtype=torch.float32)
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    return TensorDataset(X, y)