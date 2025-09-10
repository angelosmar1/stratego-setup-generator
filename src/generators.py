import torch
from torch import nn
import torch.nn.functional as F

from .constants import START, NUM_SETUP_SQUARES, PIECE_COUNTS, NUM_PIECE_TYPES


class LSTMGenerator(nn.Module):

    def __init__(self, hidden_size, embedding_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(NUM_PIECE_TYPES+1, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, NUM_PIECE_TYPES)

    def forward(self, x, hidden=None):
        # x shape should be (batch_size, seq_len) and dtype should be long
        x = self.embedding(x)   # (batch_size, seq_len, embedding_dim)
        x, hidden = self.lstm(x, hidden)   # (batch_size, seq_len, hidden_size)
        x = self.fc(x)   # (batch_size, seq_len, num_outputs)
        x = x.permute(0, 2, 1)   # for compatibility with nn.CrossEntropyLoss
        return x, hidden

    @torch.no_grad()
    def generate_setups(self, num_setups=1, seed=None):
        device = next(self.parameters()).device
        self.eval()
        setups = torch.zeros((num_setups, NUM_SETUP_SQUARES), dtype=torch.uint8)
        counts = torch.tensor(list(PIECE_COUNTS.values())).repeat(num_setups, 1).to(device)
        positions = torch.arange(num_setups)
        hidden = None
        new_piece = torch.tensor([START]).repeat(num_setups, 1).to(device)
        distributions = []
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        else:
            rng.seed()
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
        return setups.cpu().numpy(), distributions.cpu().numpy()


class TransformerGenerator(nn.Module):

    def __init__(self, embedding_dim, num_layers, num_heads, dropout=0.0):
        super(TransformerGenerator, self).__init__()
        self.embedding = nn.Embedding(NUM_PIECE_TYPES + 1, embedding_dim)
        self.pos_embedding = nn.Embedding(NUM_SETUP_SQUARES, embedding_dim)
        self.layers = nn.ModuleList([
              TransformerBlock(embedding_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embedding_dim, NUM_PIECE_TYPES)
        mask_size = (NUM_SETUP_SQUARES, NUM_SETUP_SQUARES)
        self.register_buffer("mask", torch.ones(mask_size).triu(diagonal=1).bool())

    def forward(self, x):
        # x shape should be (batch_size, seq_len) and dtype should be long
        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)   # (batch_size, seq_len, embedding_dim)
        for layer in self.layers:
            x = layer(x, self.mask[:x.size(1), :x.size(1)])
        x = self.fc_out(x)   # (batch_size, seq_len, num_outputs)
        x = x.permute(0, 2, 1)   # for compatibility with nn.CrossEntropyLoss
        return x

    @torch.no_grad()
    def generate_setups(self, num_setups=1, seed=None, batch_size=256):

        device = next(self.parameters()).device
        self.eval()
        setups = []
        distributions = []
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        else:
            rng.seed()

        for batch_start in range(0, num_setups, batch_size):
            cur_batch_size = min(batch_size, num_setups - batch_start)
            positions = torch.arange(cur_batch_size)
            cur_setups = torch.zeros((cur_batch_size, NUM_SETUP_SQUARES + 1), dtype=torch.uint8)
            cur_setups[:, 0] = torch.tensor([START]).repeat(cur_batch_size, 1).squeeze(-1)
            cur_distributions = []
            counts = torch.tensor(list(PIECE_COUNTS.values())).repeat(cur_batch_size, 1).to(device)

            for square in range(1, NUM_SETUP_SQUARES + 1):
                out = self(cur_setups[:, :square].long().to(device))    # (num_setups, num_outputs, seq_len)
                out = out[:, :, -1].squeeze(-1)      # (num_setups, num_outputs)
                out.masked_fill_(counts==0, float("-inf"))
                distribution = F.softmax(out, dim=-1)     # (num_setups, num_outputs)
                cur_distributions.append(distribution)
                new_piece = torch.multinomial(distribution, 1, generator=rng)
                counts[positions, new_piece.squeeze(-1)] -= 1
                cur_setups[:, square] = new_piece.squeeze(-1)

            setups.append(cur_setups[:, 1:].cpu())
            distributions.append(torch.stack(cur_distributions).transpose(0, 1).cpu())

        setups = torch.cat(setups, dim=0).numpy()
        distributions = torch.cat(distributions, dim=0).numpy()
        return setups, distributions


class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, ffn_dim_multiplier=4, dropout=0.0):
        super(TransformerBlock, self).__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim_multiplier * embedding_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim_multiplier * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.layer_norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        return x
