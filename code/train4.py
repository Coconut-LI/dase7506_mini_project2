"""Train the my_model4 scorer (comparison heads + hand-crafted soft features).

Identical selection loss, loop and heads to train3.py; the only change is that
each candidate's input carries two extra soft features from my_model4.features:
an arithmetic-error signal (count by default, 1/0 with --calc-binary) and a
final-answer consistency flag. Use --no-calc / --no-consistency to drop either
for a direct ablation.
"""
import argparse
import json
import os
from pathlib import Path
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'

import numpy as np
import torch

from common import DATA, check_data, load_inputs, parse_answer, read_jsonl
from my_model4 import build_model, features


def labels(bank, split, data_dir):
    check_data(data_dir, (split,))
    references = read_jsonl(data_dir / f'gsm8k_{split}.jsonl')
    if len(references) != len(bank):
        raise ValueError('Reference length mismatch')

    result = []
    for row, gold in zip(bank, references):
        if row['id'] != gold['id'] or row['question'] != gold['question']:
            raise ValueError('Reference alignment mismatch')
        answer = parse_answer(gold['answer'])
        if answer is None:
            raise ValueError('Unparseable reference')
        result.append([parse_answer(c['text']) == answer for c in row['candidates']])

    y = np.asarray(result, dtype=bool)
    if not y.any(axis=1).all():
        raise ValueError('A question has no correct candidate')
    return y


def selection_loss(scores, targets):
    # scores: [batch, 4]
    # targets: [batch, 4] bool
    all_term = torch.logsumexp(scores, dim=1)
    positive_scores = scores.masked_fill(~targets, float('-inf'))
    positive_term = torch.logsumexp(positive_scores, dim=1)
    return (all_term - positive_term).mean()


@torch.no_grad()
def accuracy(model, x, y, batch_size=1024):
    model.eval()
    correct = 0
    total = x.size(0)

    for begin in range(0, total, batch_size):
        xb = x[begin:begin + batch_size]
        yb = y[begin:begin + batch_size]
        selected = model(xb).argmax(dim=1)
        correct += yb[torch.arange(len(selected)), selected].float().sum().item()

    return correct / total


@torch.no_grad()
def evaluate(model, x, y, batch_size=1024):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for begin in range(0, x.size(0), batch_size):
        xb = x[begin:begin + batch_size]
        yb = y[begin:begin + batch_size]
        loss = selection_loss(model(xb), yb)
        batch_n = xb.size(0)
        total_loss += loss.item() * batch_n
        total_count += batch_n

    return total_loss / total_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/set_calc'))
    parser.add_argument('--model', choices=('deepset', 'attention'), default='deepset',
                        help='Candidate-comparison head')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability for the comparison head')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-vote', action='store_true', help='Embedding-only ablation')
    parser.add_argument('--no-calc', action='store_true', help='Drop the arithmetic-error feature')
    parser.add_argument('--calc-binary', action='store_true',
                        help='Use a 1/0 "has error" calc feature instead of the error count')
    parser.add_argument('--no-consistency', action='store_true', help='Drop the final-answer consistency feature')
    args = parser.parse_args()

    if args.hidden_dim < 1 or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error('hidden-dim/epochs/batch-size/lr must be positive; weight-decay must be nonnegative')
    if not 0.0 <= args.dropout < 1.0:
        parser.error('dropout must be in [0, 1)')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory to preserve earlier experiments')

    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    started = time.perf_counter()
    inputs = {}
    use_vote = not args.no_vote
    use_calc = not args.no_calc
    calc_binary = args.calc_binary
    use_consistency = not args.no_consistency

    for split in ('train', 'validation'):
        bank, arrays = load_inputs(split, args.data_dir)
        x_np = features(bank, arrays, use_vote=use_vote, use_calc=use_calc,
                        calc_binary=calc_binary, use_consistency=use_consistency)
        y_np = labels(bank, split, args.data_dir)
        inputs[split] = (
            torch.from_numpy(x_np).float(),
            torch.from_numpy(y_np).bool(),
        )

    x, y = inputs['train']
    vx, vy = inputs['validation']

    model = build_model(input_dim=x.shape[-1], hidden_dim=args.hidden_dim, model_type=args.model, dropout=args.dropout)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best = -1.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(x.size(0))

        for begin in range(0, x.size(0), args.batch_size):
            indices = order[begin:begin + args.batch_size]
            xb = x[indices]
            yb = y[indices]

            optimizer.zero_grad()
            loss = selection_loss(model(xb), yb)
            loss.backward()
            optimizer.step()

        train_loss = evaluate(model, x, y)
        val_loss = evaluate(model, vx, vy)
        train_accuracy = accuracy(model, x, y)
        val_accuracy = accuracy(model, vx, vy)

        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'validation_loss': val_loss,
            'train_accuracy': 100 * train_accuracy,
            'validation_accuracy': 100 * val_accuracy,
        }
        history.append(record)

        if val_accuracy > best:
            best = val_accuracy
            best_epoch = epoch
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'input_dim': int(x.shape[-1]),
                    'hidden_dim': int(args.hidden_dim),
                    'use_vote': bool(use_vote),
                    'use_calc': bool(use_calc),
                    'calc_binary': bool(calc_binary),
                    'use_consistency': bool(use_consistency),
                    'model_type': args.model,
                    'dropout': args.dropout,
                },
                args.output_dir / 'model.pt',
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(record), flush=True)

    summary = {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'numpy_version': np.__version__,
        'torch_version': torch.__version__,
        'best_epoch': best_epoch,
        'best_validation_accuracy': 100 * best,
        'training_seconds': time.perf_counter() - started,
        'checkpoint_bytes': (args.output_dir / 'model.pt').stat().st_size,
        'history': history,
    }
    (args.output_dir / 'training.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Best validation accuracy: {100 * best:.2f}% at epoch {best_epoch}')


if __name__ == '__main__':
    main()
