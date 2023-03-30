import torch
from torch import Tensor
import torch.nn as nn
from .attention import MultiHeadAttention
from .residual import ResidualConnection
from .ffn import PositionWiseFeedForward

from typing import Callable


class FFT(nn.Module):
    def __init__(self, d_model: int, heads: int, hidden_dim: int, dropout_rate: float, eps: float, activation: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(heads=heads, d_model=d_model)
        self.ffn = PositionWiseFeedForward(channels=d_model, hidden_channels=hidden_dim, activation=activation)

        self.residual_connection_1 = ResidualConnection(d_model=d_model, dropout_rate=dropout_rate, eps=eps)
        self.residual_connection_2 = ResidualConnection(d_model=d_model, dropout_rate=dropout_rate, eps=eps)

    def forward(self, x: Tensor, mask: Tensor):
        """ 
            x: (batch_size, length, d_model) 
        """
        q = k = v = x
        attention_output = self.multi_head_attention(q, k, v, mask)
        sub_layer_1 = self.residual_connection_1(attention_output, x)

        residual = sub_layer_1
        sub_layer_1 = torch.transpose(sub_layer_1, -1, -2)
        ffn_output = self.ffn(sub_layer_1)
        ffn_output = torch.transpose(ffn_output, -1, -2)
        sub_layer_2 = self.residual_connection_2(ffn_output, residual)

        return sub_layer_2

