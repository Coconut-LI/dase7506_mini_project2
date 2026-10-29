"""PyTorch two-layer candidate scorer with the official predict() interface."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import parse_answer

"""输入候选题目列表、特征数组和是否启用投票特征，先归一化候选嵌入并可选拼接投票值，输出每个候选的特征张量。"""
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

"""输入特征维度和隐藏层维度，构建两层线性加激活的候选打分网络，输出初始化完成的模型实例。"""
class CandidateMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    """输入形状为批大小乘候选数乘特征维度的张量，逐候选计算网络分数并压缩最后一维，输出每题四个候选的分数张量。"""
    def forward(self, x):
        # x: [batch, 4, dim]
        scores = self.net(x).squeeze(-1)  # [batch, 4]
        return scores

"""输入特征维度和隐藏层维度，实例化候选打分 MLP，输出可训练的模型对象。"""
def build_model(input_dim, hidden_dim):
    return CandidateMLP(input_dim=input_dim, hidden_dim=hidden_dim)

"""输入检查点文件路径，读取模型配置与参数并恢复到 CPU 推理状态，输出模型、是否使用投票特征和输入维度。"""
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(Path(checkpoint_path), map_location='cpu')
    input_dim = int(checkpoint['input_dim'])
    hidden_dim = int(checkpoint['hidden_dim'])
    use_vote = bool(checkpoint['use_vote'])
    model = build_model(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, use_vote, input_dim

"""输入候选题目列表、特征数组和检查点路径，恢复模型后计算各候选分数并选最大值，输出每道题预测候选的索引列表。"""
@torch.no_grad()
def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Pass --checkpoint runs/mlp_torch/model.pt')

    model, use_vote, input_dim = load_checkpoint(checkpoint)
    x = features(bank, arrays, use_vote=use_vote)

    if x.shape[-1] != input_dim:
        raise ValueError(
            f'Checkpoint feature dimension mismatch: expected {input_dim}, got {x.shape[-1]}'
        )

    x_tensor = torch.from_numpy(x).float()
    scores = model(x_tensor)
    return scores.argmax(dim=1).tolist()