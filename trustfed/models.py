"""
models.py -- detector architectures.

All models use GroupNorm, never BatchNorm. BatchNorm keeps running statistics
in buffers that FedAvg-style weight vectors do not carry, so in the original
code every non-DP run evaluated the global model with untouched running stats.
GroupNorm has no buffers, works with Opacus per-sample gradients, and makes DP
and non-DP runs use the SAME architecture (required for a fair ablation).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(dim, groups=8):
    return nn.GroupNorm(min(groups, dim), dim)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), _gn(dim), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(dim, dim), _gn(dim))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ResidualMLP(nn.Module):
    def __init__(self, n_features, n_classes, hidden=256, n_blocks=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(n_features, hidden), _gn(hidden), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden // 2),
                                  nn.GELU(), nn.Linear(hidden // 2, n_classes))

    def forward(self, x):
        return self.head(self.blocks(self.input_proj(x)))


class CNNBiLSTMAttention(nn.Module):
    """GPU variant. Uses Opacus DPLSTM when dp=True (nn.LSTM is not DP-compatible)."""
    def __init__(self, n_features, n_classes, filters=(64, 128), lstm_hidden=128,
                 dropout=0.3, dp=False):
        super().__init__()
        self.conv1 = nn.Conv1d(1, filters[0], 3, padding=1); self.n1 = _gn(filters[0])
        self.conv2 = nn.Conv1d(filters[0], filters[1], 3, padding=1); self.n2 = _gn(filters[1])
        self.drop = nn.Dropout(dropout)
        if dp:
            from opacus.layers import DPLSTM
            self.lstm = DPLSTM(filters[1], lstm_hidden, 2, batch_first=True,
                               bidirectional=True, dropout=dropout)
        else:
            self.lstm = nn.LSTM(filters[1], lstm_hidden, 2, batch_first=True,
                                bidirectional=True, dropout=dropout)
        self.attn = nn.Linear(2 * lstm_hidden, 1)
        self.head = nn.Sequential(nn.Linear(2 * lstm_hidden, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.gelu(self.n1(self.conv1(x)))
        x = F.gelu(self.n2(self.conv2(x)))
        x = self.drop(x).permute(0, 2, 1)
        x, _ = self.lstm(x)
        a = torch.softmax(self.attn(x).squeeze(-1), dim=-1).unsqueeze(-1)
        return self.head((x * a).sum(1))


class BiLSTMClassifier(nn.Module):
    """Centralised LSTM-IDS baseline (each feature treated as a time step)."""
    def __init__(self, n_features, n_classes, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, n_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 64),
                                  nn.GELU(), nn.Linear(64, n_classes))

    def forward(self, x):
        h, _ = self.lstm(x.unsqueeze(-1))
        return self.head(h.mean(1))


def get_model(n_features, n_classes, arch="residual_mlp", hidden=256, n_blocks=3, dp=False):
    arch = arch.lower()
    if arch in ("cnn_bilstm_attn", "cnn", "paper"):
        return CNNBiLSTMAttention(n_features, n_classes, dp=dp)
    if arch in ("bilstm", "lstm_ids"):
        return BiLSTMClassifier(n_features, n_classes)
    return ResidualMLP(n_features, n_classes, hidden=hidden, n_blocks=n_blocks)


def get_weights(model):
    """Flat vector of ALL state (parameters + buffers) so nothing is left behind."""
    return np.concatenate([t.detach().cpu().numpy().ravel().astype(np.float32)
                           for t in model.state_dict().values()])


def set_weights(model, w):
    sd, off = model.state_dict(), 0
    new = {}
    for k, t in sd.items():
        n = t.numel()
        new[k] = torch.as_tensor(w[off:off + n], dtype=t.dtype).reshape(t.shape).to(t.device)
        off += n
    if off != len(w):
        raise ValueError(f"weight length {len(w)} != model size {off}")
    model.load_state_dict(new)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def predict_proba(model, X, batch=4096, device="cpu"):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
            out.append(torch.softmax(model(xb), -1).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 1))
