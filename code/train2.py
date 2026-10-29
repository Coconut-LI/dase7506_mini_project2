"""Train a two-layer PyTorch MLP candidate scorer using all correct candidates as positive targets.

Loss per question: logsumexp(all scores) - logsumexp(correct scores).
Only train references update weights; validation selects the checkpoint.
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
from my_model import build_model, features

"""输入候选题目列表、数据划分名和数据目录，读取并对齐参考答案后标记每个候选是否正确，输出布尔标签矩阵。"""
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

"""输入每题四个候选的分数张量和正确候选布尔标记，计算全体分数与正确分数的对数和差，输出批次平均选择损失。"""
def selection_loss(scores, targets):
    # scores: [batch, 4]
    # targets: [batch, 4] bool
    all_term = torch.logsumexp(scores, dim=1)
    positive_scores = scores.masked_fill(~targets, float('-inf'))
    positive_term = torch.logsumexp(positive_scores, dim=1)
    return (all_term - positive_term).mean()

"""输入模型、特征张量、标签张量和批大小，分批选出每题最高分候选并检查是否命中正确答案，输出整体准确率。"""
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

"""输入模型、特征张量、标签张量和批大小，分批计算选择损失并按样本数加权汇总，输出平均损失。"""
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

"""输入命令行参数，完成数据加载、特征构造、模型训练与验证选优并保存结果，输出训练日志、检查点和汇总文件。"""
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/mlp_torch'))
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-vote', action='store_true', help='Embedding-only ablation')
    args = parser.parse_args()

    if args.hidden_dim < 1 or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error('hidden-dim/epochs/batch-size/lr must be positive; weight-decay must be nonnegative')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory to preserve earlier experiments')

    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    started = time.perf_counter()
    inputs = {}
    use_vote = not args.no_vote

    for split in ('train', 'validation'):
        bank, arrays = load_inputs(split, args.data_dir)
        x_np = features(bank, arrays, use_vote=use_vote)
        y_np = labels(bank, split, args.data_dir)
        inputs[split] = (
            torch.from_numpy(x_np).float(),
            torch.from_numpy(y_np).bool(),
        )

    x, y = inputs['train']
    vx, vy = inputs['validation']

    model = build_model(input_dim=x.shape[-1], hidden_dim=args.hidden_dim)
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