"""Candidate scorer with hand-crafted soft features (my_model4).

Same permutation-equivariant comparison heads as my_model3 (DeepSets /
self-attention), but each candidate's input is augmented with two extra
features: the number of arithmetic errors in its reasoning, and whether the
reasoning chain's last computed value disagrees with its stated final answer.

This is a *soft* signal the model can learn to downweight, not a hard filter:
a candidate is never removed outright (removing could delete the only correct
candidate, or leave a question with no choice at all). The model decides how
much, if at all, an arithmetic slip should reduce a candidate's score.

The arithmetic check is the same deterministic heuristic as analyze_calc.py:
every explicit ``X = Y`` / ``X ≈ Y`` equation is evaluated with a safe
arithmetic evaluator and flagged when the stated right-hand side disagrees with
the computed value. It reads only the candidate text, never the gold answer,
so it is legal at both training and inference time and costs nothing in
inference assets (no learned weights).
"""
import ast
from collections import Counter
from fractions import Fraction
import operator
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn

from common import parse_answer

# --- safe arithmetic evaluator (numbers and + - * / **, parentheses) ---
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError('non-numeric constant')
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    raise ValueError('unsupported node')


def safe_eval(expr):
    return _eval_node(ast.parse(expr, mode='eval'))


def normalize(text):
    """Flatten markdown/LaTeX/unicode into plain arithmetic text."""
    t = str(text)
    t = re.sub(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'(\1)/(\2)', t)
    t = re.sub(r'\\(?:text|mathrm|mathbf|mathit|mbox)\s*\{[^{}]*\}', '', t)
    t = t.replace('\\left', '').replace('\\right', '')
    t = t.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/')
    t = t.replace('\\approx', '\u2248').replace('\\boxed', '').replace('\\box', '')
    t = re.sub(r'\\[a-zA-Z]+', '', t)
    t = t.replace('\u00d7', '*').replace('\u00f7', '/').replace('\u00b7', '*').replace('\u2212', '-')
    t = t.replace('$', '')                 # currency and LaTeX math delimiters
    t = re.sub(r'\*\*', '', t)             # markdown bold
    t = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', t)  # thousands separators
    return t


_EQ_OP = re.compile(r'(=|≈)')


def extract_equations(text, eq_tol=1e-3, approx_tol=1e-2):
    """Return every explicit ``X = Y`` / ``X ≈ Y`` equation with its match flag."""
    t = normalize(text)
    found = []
    for m in _EQ_OP.finditer(t):
        op = m.group(1)
        pos = m.start()
        # Left-hand side: walk left over arithmetic characters.
        j = pos - 1
        while j >= 0 and t[j] in '0123456789.+-*/() ':
            j -= 1
        lhs = t[j + 1:pos].strip()
        # Right-hand side: leading (signed) number.
        rm = re.match(r'\s*([-+]?\d[\d.,]*)', t[pos + 1:])
        rhs = rm.group(1) if rm else ''
        # A "calculation" needs a real expression, not a bare number.
        if not lhs or not rhs or not any(ch in lhs for ch in '+-*/('):
            continue
        # LHS must start with a digit or '('; a leading '+/-' means the walk
        # hit a variable/word and truncated the expression (e.g. "a + 24 = ...").
        if lhs[0] not in '0123456789(':
            continue
        # RHS must be a complete number, not a link in a longer chain
        # (e.g. "8 - 6 = 2 + 0" or "= 8 - 6 = 2").
        after = pos + 1 + rm.end()
        while after < len(t) and t[after] == ' ':
            after += 1
        if after < len(t) and t[after] in '+-*/=≈(':
            continue
        try:
            computed = safe_eval(lhs)
            stated = float(rhs.replace(',', ''))
        except Exception:
            continue
        tol = approx_tol if op == '≈' else eq_tol
        matched = abs(computed - stated) <= tol * max(1.0, abs(computed), abs(stated))
        found.append({
            'lhs': lhs,
            'op': op,
            'rhs': rhs,
            'computed': round(computed, 6),
            'stated': stated,
            'matched': bool(matched),
        })
    return found


def count_calc_errors(text, eq_tol=1e-3, approx_tol=1e-2):
    """Number of arithmetic errors detected in a candidate's reasoning text."""
    return sum(1 for q in extract_equations(text, eq_tol, approx_tol) if not q['matched'])


def final_answer_mismatch(text, tol=1e-3):
    """1.0 if the reasoning chain's last computed value disagrees with the
    candidate's own stated final answer, else 0.0 (0 when either is missing).

    The chain's conclusion is the right-hand side of the last explicit ``X = Y``
    equation; the final answer is the parsed `####` (or boxed) value. A
    candidate whose chain computes 14 but whose boxed answer says 13 is
    internally inconsistent and a strong 'likely wrong' signal.
    """
    final = parse_answer(text)
    equations = extract_equations(text)
    if final is None or not equations:
        return 0.0
    stated = equations[-1]['stated']
    try:
        final_value = float(Fraction(final))
    except (ValueError, ZeroDivisionError):
        return 0.0
    matched = abs(final_value - stated) <= tol * max(1.0, abs(final_value), abs(stated))
    return 0.0 if matched else 1.0


def features(bank, arrays, use_vote=True, use_calc=True, calc_binary=False, use_consistency=True):
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
    if use_calc:
        # Soft signal: the number of arithmetic slips, or (calc_binary) a 1/0
        # flag for "has at least one slip".
        calc = []
        for row in bank:
            n_errors = [count_calc_errors(c['text']) for c in row['candidates']]
            if calc_binary:
                calc.append([1.0 if n > 0 else 0.0 for n in n_errors])
            else:
                calc.append([float(n) for n in n_errors])
        calc_array = np.asarray(calc, dtype=np.float32)[..., None]
        x = np.concatenate([x, calc_array], axis=-1)
    if use_consistency:
        # Soft signal: 1 if the chain's last computed value disagrees with the
        # candidate's own stated final answer, else 0.
        consistency = []
        for row in bank:
            consistency.append([final_answer_mismatch(c['text'])
                                for c in row['candidates']])
        consistency_array = np.asarray(consistency, dtype=np.float32)[..., None]
        x = np.concatenate([x, consistency_array], axis=-1)
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
    use_calc = bool(checkpoint.get('use_calc', False))
    calc_binary = bool(checkpoint.get('calc_binary', False))
    use_consistency = bool(checkpoint.get('use_consistency', False))
    model_type = str(checkpoint['model_type'])
    dropout = float(checkpoint.get('dropout', 0.1))
    model = build_model(input_dim=input_dim, hidden_dim=hidden_dim, model_type=model_type, dropout=dropout)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, use_vote, use_calc, calc_binary, use_consistency, input_dim


@torch.no_grad()
def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Pass --checkpoint runs/set_calc/model.pt')

    model, use_vote, use_calc, calc_binary, use_consistency, input_dim = load_checkpoint(checkpoint)
    x = features(bank, arrays, use_vote=use_vote, use_calc=use_calc, calc_binary=calc_binary, use_consistency=use_consistency)

    if x.shape[-1] != input_dim:
        raise ValueError(
            f'Checkpoint feature dimension mismatch: expected {input_dim}, got {x.shape[-1]}'
        )

    x_tensor = torch.from_numpy(x).float()
    scores = model(x_tensor)
    return scores.argmax(dim=1).tolist()
