import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
class VariancePredictor(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=d_model, out_channels=hidden_dim, kernel_size=3, padding=1, stride=1)
        self.conv1d_2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1, stride=1)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=d_model)

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=hidden_dim, eps=eps)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=hidden_dim, eps=eps)
        self.dropout_layer_1 = nn.Dropout(p=dropout_rate)
        self.dropout_layer_2 = nn.Dropout(p=dropout_rate)

        self.to(device)

    def forward(self, x: Tensor, mask: Tensor = None):
        """ 
            x: (batch_size, length, d_model) 
        """
        x = x.transpose(-1, -2)
        x = self.conv1d_1(x)
        x = F.relu(x) # (batch_size, hidden_dim, length)

        x = x.transpose(-1, -2)
        x = self.layer_norm_1(x)
        x = self.dropout_layer_1(x) # (batch_size, length, hidden_dim)

        x = x.transpose(-1, -2)
        x = self.conv1d_2(x)
        x = F.relu(x) # (batch_size, hidden_dim, length)

        x = x.transpose(-1, -2)
        x = self.layer_norm_2(x)
        x = self.dropout_layer_2(x) # (batch_size, length, hidden_dim)

        x = self.linear(x)
        x = x.squeeze(-1)
        if mask is not None:
            x = x.masked_fill(mask==1, 0.0)
        return x

