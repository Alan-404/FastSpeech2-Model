import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from typing import Callable
from model.utils.net import PostNet

from model.utils.fft import FFT


class Decoder(nn.Module):
    def __init__(self, n: int, d_model: int, heads: int, hidden_dim: int, dropout_rate: float, eps: float, n_mels: int, kernel_size: int, num_layers: int, activation: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.layers = nn.Module([FFT(d_model=d_model, heads=heads, hidden_dim=hidden_dim, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)])
        self.linear = nn.Linear(in_features=d_model, out_features=n_mels)
        self.post_net = PostNet(n_mels=n_mels, d_model=d_model, kernel_size=kernel_size, num_layers=num_layers)
    def forward(self, x: Tensor, mask: Tensor):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.linear(x)
        x = self.post_net(x)
        return x