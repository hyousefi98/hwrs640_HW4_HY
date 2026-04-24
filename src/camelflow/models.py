from __future__ import annotations
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, n_dyn: int, n_static: int, hidden: int = 128,
                 num_layers: int = 1, dropout: float = 0.4):
        super().__init__()
        self.kwargs = dict(n_dyn=n_dyn, n_static=n_static, hidden=hidden,
                           num_layers=num_layers, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=n_dyn + n_static,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_dyn: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_dyn.shape
        s = x_static.unsqueeze(1).expand(B, T, -1)
        x = torch.cat([x_dyn, s], dim=-1)
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1])
        return self.head(h).squeeze(-1)


def build_model(name: str, **kw) -> nn.Module:
    if name.lower() == "lstm":
        return LSTMRegressor(**kw)
    raise ValueError(f"unknown model: {name}")
