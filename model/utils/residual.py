import torch
from torch import Tensor
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)

        self.to(device)

    def forward(self, x: Tensor, pre_x: Tensor) -> Tensor:
        x = self.dropout(x)
        x = x + pre_x
        x = self.layer_norm(x)

        return x