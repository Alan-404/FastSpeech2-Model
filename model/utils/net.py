import torch
from torch import Tensor
import torch.nn as nn


class PostNet(nn.Module):
    def __init__(self, n_mels: int, d_model: int, kernel_size: int, num_layers: int) -> None:
        super().__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            ConvNorm(
                in_channels=n_mels,
                out_channels=d_model,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1)/2),
                dilation=1
            )
        )

        for _ in range(num_layers-1):
            self.convolutions.append(
                ConvNorm(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=int((kernel_size - 1)/2),
                    dilation=1
                )
            )

        self.convolutions.append(
            ConvNorm(
                in_channels=d_model,
                out_channels=n_mels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1)/2),
                dilation=1
            )
        )

    



class ConvNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int=None, dilation: int=1) -> None:
        super().__init__()
        if padding is None:
            assert kernel_size%2==1
            padding = int(dilation * (kernel_size-1)/2)
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True
        )

        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor):
        x = self.conv1d(x)
        x = self.batch_norm(x)

        return x
        