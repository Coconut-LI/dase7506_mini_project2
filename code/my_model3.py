"""Candidate scorer that compares all four candidates instead of scoring each in isolation.

Two permutation-equivariant heads over the same per-candidate features:

  - SetScorer (deepset):  per-candidate encoder -> global summary -> score(candidate, summary).
  - SetAttentionScorer (attention): one self-attention layer over the four candidates.

Both read the same features() as my_model.py (whole-trace embeddings, optionally + vote),
so the only difference from my_model.py is the head architecture. The selection loss and
training loop in train3.py are unchanged from train2.py.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import parse_answer


def features(bank, arrays, use_vote=True):
    x = np.asarray(arrays['embeddings'], dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
    if use_vote:
        votes = []
        for row in bank:
            answers = [parse_answer(c['text']) for c in row['candidates']]
            counts = Counter(a for a in answers if a is not None)
            votes.append([counts[a] / 4 if a is not None else 0.0 for a in answers])
        vote_array = np.asarray(votes, dtype=np.float32)[..., None]
        x = np.concatenate([x, vote_array], axis=-1)
    return x


class SetScorer(nn.Module):
    """DeepSets: score each candidate conditioned on a global summary of all four.

    The per-candidate encoder is shared, and the global summary is a symmetric
    (mean) aggregate, so permuting the candidates permutes the scores identically.
    """

    def __init__(self, input_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: [batch, 4, dim]
        h = self.encoder(x)                    # [batch, 4, hidden]
        g = h.mean(dim=1, keepdim=True)        # [batch, 1, hidden]
        g = g.expand(-1, x.size(1), -1)        # [batch, 4, hidden]
        combined = torch.cat([h, g], dim=-1)   # [batch, 4, 2*hidden]
        return self.scorer(combined).squeeze(-1)  # [batch, 4]


class SetAttentionScorer(nn.Module):
    """One self-attention layer over the four candidates; no positional encoding."""

    def __init__(self, input_dim, hidden_dim=128, heads=4, dropout=0.1):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError('hidden_dim must be divisible by heads')
        self.proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, 4, dim]
        h = self.proj(x)                    # [batch, 4, hidden]
        h = self.encoder(h)                 # [batch, 4, hidden]
        return self.head(h).squeeze(-1)     # [batch, 4]


def build_model(input_dim, hidden_dim, model_type='deepset', dropout=0.1):
    if model_type == 'deepset':
        return SetScorer(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if model_type == 'attention':
        return SetAttentionScorer(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f'Unknown model_type: {model_type}')


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(Path(checkpoint_path), map_location='cpu')
    input_dim = int(checkpoint['input_dim'])
    hidden_dim = int(checkpoint['hidden_dim'])
    use_vote = bool(checkpoint['use_vote'])
    model_type = str(checkpoint['model_type'])
    dropout = float(checkpoint.get('dropout', 0.1))
    model = build_model(input_dim=input_dim, hidden_dim=hidden_dim, model_type=model_type, dropout=dropout)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, use_vote, input_dim


@torch.no_grad()
def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Pass --checkpoint runs/set/model.pt')

    model, use_vote, input_dim = load_checkpoint(checkpoint)
    x = features(bank, arrays, use_vote=use_vote)

    if x.shape[-1] != input_dim:
        raise ValueError(
            f'Checkpoint feature dimension mismatch: expected {input_dim}, got {x.shape[-1]}'
        )

    x_tensor = torch.from_numpy(x).float()
    scores = model(x_tensor)
    return scores.argmax(dim=1).tolist()
