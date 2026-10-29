"""Answer-only MLP candidate scorer; identical architecture to my_model.py, but the
input features are answer-only (no embeddings).

This is the "does not read the reasoning" arm of the ablation with the SAME model
architecture as the embedding model, so the only difference between the two arms is
the input features (2-dim parsed-answer features vs 1024-dim whole-trace embeddings).
"""
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import parse_answer


def features(bank, arrays):
    """Build per-candidate answer-only features, shape [N, 4, 2].

    `arrays` is accepted for interface parity with the embedding models but is
    deliberately ignored: reading the reasoning means reading embeddings, and
    this model must not. `parse_answer` canonicalises equivalent fractions and
    decimals, so agreement counts answer values that mean the same number.

    Feature 0 (agreement): fraction of the four candidates whose parsed final
        answer equals this candidate's answer (0.0 when unparseable).
    Feature 1 (is_none): 1.0 when this candidate's final answer is unparseable,
        else 0.0.
    """
    feats = []
    for row in bank:
        answers = [parse_answer(c['text']) for c in row['candidates']]
        counts = Counter(a for a in answers if a is not None)
        feats.append([
            [counts[a] / 4.0, 0.0] if a is not None else [0.0, 1.0]
            for a in answers
        ])
    return np.asarray(feats, dtype=np.float32)


class CandidateMLP(nn.Module):
    """Two-layer MLP identical to my_model.CandidateMLP; scores each candidate."""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: [batch, 4, dim]
        return self.net(x).squeeze(-1)  # [batch, 4]


def build_model(input_dim, hidden_dim):
    return CandidateMLP(input_dim=input_dim, hidden_dim=hidden_dim)


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(Path(checkpoint_path), map_location='cpu')
    input_dim = int(checkpoint['input_dim'])
    hidden_dim = int(checkpoint['hidden_dim'])
    model = build_model(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, input_dim


@torch.no_grad()
def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Pass --checkpoint runs/answer_only_mlp/model.pt')

    model, input_dim = load_checkpoint(checkpoint)
    x = features(bank, arrays)

    if x.shape[-1] != input_dim:
        raise ValueError(
            f'Checkpoint feature dimension mismatch: expected {input_dim}, got {x.shape[-1]}'
        )

    x_tensor = torch.from_numpy(x).float()
    scores = model(x_tensor)
    return scores.argmax(dim=1).tolist()
