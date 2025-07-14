import torch
from torch import nn
import torch.nn.functional as F

from .stratego_utils import START, NUM_SETUP_SQUARES, PIECE_COUNTS, NUM_PIECE_TYPES


class LSTMGenerator(nn.Module):

    def __init__(self, hidden_size, embedding_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(NUM_PIECE_TYPES+1, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, NUM_PIECE_TYPES)

    def forward(self, x, hidden=None):
        # x shape should be (batch_size, seq_len)
        x = self.embedding(x)   # (batch_size, seq_len, embedding_dim)
        x, hidden = self.lstm(x, hidden)   # (batch_size, seq_len, hidden_size)
        x = self.fc(x)   # (batch_size, seq_len, num_outputs)
        x = x.permute(0, 2, 1)   # for compatibility with nn.CrossEntropyLoss
        return x, hidden
        
    def generate_setups(self, num_setups=1, seed=None):
        device = next(self.parameters()).device
        self.eval()
        setups = torch.zeros((num_setups, NUM_SETUP_SQUARES), dtype=torch.long).to(device)
        counts = torch.tensor(list(PIECE_COUNTS.values())).repeat(num_setups, 1).to(device)
        positions = torch.arange(num_setups)
        hidden = None
        new_piece = torch.tensor([START]).repeat(num_setups, 1).to(device)
        distributions = []
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        with torch.no_grad():
            for square in range(NUM_SETUP_SQUARES):
                out, hidden = self(new_piece, hidden)   # (num_setups, num_outputs, 1)
                out = out.squeeze(-1)    # (num_setups, num_outputs)
                out.masked_fill_(counts==0, float("-inf"))
                distribution = F.softmax(out, dim=-1)   # (num_setups, num_outputs)
                distributions.append(distribution)
                new_piece = torch.multinomial(distribution, 1, generator=rng)
                counts[positions, new_piece.squeeze(-1)] -= 1
                setups[:, square] = new_piece.squeeze(-1)
        distributions = torch.stack(distributions).transpose(0, 1)
        return setups.cpu(), distributions.cpu()
