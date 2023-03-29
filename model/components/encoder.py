import torch
from torch import Tensor
import torch.nn as nn
from model.utils.fft import FFT

from typing import Callable


class Encoder(nn.Module):
    def __init__(self, n: int, d_model: int, heads: int, hidden_dim: int, dropout_rate: float, eps: float, activation: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FFT(d_model=d_model, heads=heads, hidden_dim=hidden_dim, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)])

    def forward(self, x: Tensor, mask: Tensor):
        for layer in self.layers:
            x = layer(x, mask)

        return x