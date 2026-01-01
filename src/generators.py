import torch
from torch import nn
import torch.nn.functional as F

from .constants import START, NUM_SETUP_SQUARES, PIECE_COUNTS, NUM_PIECE_TYPES


class SetupGenerator(nn.Module):

    @torch.no_grad()
    def generate_setups(self, num_setups, seed, batch_size):

        self.eval()
        device = next(self.parameters()).device

        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        else:
            rng.seed()

        setups = []
        distributions = []

        for batch_start in range(0, num_setups, batch_size):
            cur_batch_size = min(batch_size, num_setups - batch_start)
            positions = torch.arange(cur_batch_size)
            setups_batch = torch.empty((cur_batch_size, NUM_SETUP_SQUARES + 1), dtype=torch.long)
            setups_batch[:, 0] = START
            distributions_batch = []
            counts = torch.tensor(list(PIECE_COUNTS.values()), device=device).repeat(cur_batch_size, 1)
            model_state = None

            for square in range(1, NUM_SETUP_SQUARES + 1):
                logits, model_state = self._get_logits_next(setups_batch[:, :square].to(device), model_state)
                logits.masked_fill_(counts==0, float("-inf"))
                distribution = F.softmax(logits, dim=-1)
                new_piece = torch.multinomial(distribution, 1, generator=rng).squeeze(-1)
                counts[positions, new_piece] -= 1
                setups_batch[:, square] = new_piece
                distributions_batch.append(distribution)

            setups.append(setups_batch[:, 1:].to(device="cpu", dtype=torch.uint8))
            distributions.append(torch.stack(distributions_batch, dim=1).cpu())

        setups = torch.cat(setups).numpy()
        distributions = torch.cat(distributions).numpy()

        return setups, distributions


class LSTMGenerator(SetupGenerator):

    def __init__(self, hidden_size, embedding_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(NUM_PIECE_TYPES + 1, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, NUM_PIECE_TYPES)

    def forward(self, x, hidden_state=None):
        # x shape should be (batch_size, seq_len) and dtype should be long
        x = self.embedding(x)   # (batch_size, seq_len, embedding_dim)
        x, hidden_state = self.lstm(x, hidden_state)   # (batch_size, seq_len, hidden_size)
        x = self.fc(x)   # (batch_size, seq_len, NUM_PIECE_TYPES)
        x = x.permute(0, 2, 1)   # for compatibility with nn.CrossEntropyLoss
        return x, hidden_state

    def _get_logits_next(self, x, hidden_state):
        logits, hidden_state = self(x[:, -1:], hidden_state)
        return logits[:, :, -1], hidden_state

    def generate_setups(self, num_setups=1, seed=None, batch_size=10_000):
        return super().generate_setups(num_setups, seed, batch_size)


class TransformerGenerator(SetupGenerator):

    def __init__(self, embedding_dim, num_layers, num_heads, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(NUM_PIECE_TYPES + 1, embedding_dim)
        self.pos_embedding = nn.Embedding(NUM_SETUP_SQUARES, embedding_dim)
        self.layers = nn.ModuleList([
              TransformerBlock(embedding_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embedding_dim, NUM_PIECE_TYPES)
        mask_size = (NUM_SETUP_SQUARES, NUM_SETUP_SQUARES)
        self.register_buffer("mask", torch.ones(mask_size, dtype=torch.bool).triu(diagonal=1))

    def forward(self, x):
        # x shape should be (batch_size, seq_len) and dtype should be long
        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)   # (batch_size, seq_len, embedding_dim)
        for layer in self.layers:
            x = layer(x, self.mask[:x.size(1), :x.size(1)])
        x = self.fc_out(x)   # (batch_size, seq_len, NUM_PIECE_TYPES)
        x = x.permute(0, 2, 1)   # for compatibility with nn.CrossEntropyLoss
        return x

    def _get_logits_next(self, x, dummy_state=None):
        return self(x)[:, :, -1], None

    def generate_setups(self, num_setups=1, seed=None, batch_size=256):
        return super().generate_setups(num_setups, seed, batch_size)


class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, ffn_dim_multiplier=4, dropout=0.0):
        super().__init__()
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
