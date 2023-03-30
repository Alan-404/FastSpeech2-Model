import torch
from torch import Tensor
import torch.nn as nn

from typing import Callable

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class PositionWiseFeedForward(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, activation: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=hidden_channels, out_channels=channels, kernel_size=3, stride=1, padding=1)

        self.activation = activation

        self.to(device)

    def forward(self, x: Tensor):
        print(x.size())
        x = self.conv1d_1(x)
        x = self.activation(x)
        x = self.conv1d_2(x)
        x = self.activation(x)

        return x