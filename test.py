#%%
import torch
from typing import Callable
from torch import Tensor
import numpy as np
# %%
phoneme_size: int = 1000
n: int = 6
d_model: int = 256
heads: int = 8
hidden_dim: int = 2048
n_mels: int = 80
n_bins: int = 120
kernel_size: int = 5
num_layers: int = 5
pitch_feature_level: str = "phoneme_level"
energy_feature_level: str = "phoneme_level"
pitch_quantization: str = "log"
energy_quantization: str = "log"
dropout_rate: float = 0.1
eps: float = 0.1
activation: Callable[[Tensor], Tensor] = torch.nn.functional.relu
from model.components.encoder import Encoder
from model.components.adaptor import VarianceAdaptor

# %%
encoder = Encoder(n=n, d_model=d_model, heads=heads, hidden_dim=hidden_dim, dropout_rate=dropout_rate, eps=eps, activation=activation)
# %%
a = torch.rand((10, 40, d_model)).type(torch.float32)
# %%
device = torch.device('cuda')
# %%
a = a.to(device)
#%%
b = encoder(a, None)
# %%
adaptor = VarianceAdaptor(d_model=d_model, hidden_dim=hidden_dim, n_bins=n_bins, pitch_feature_level=pitch_feature_level, energy_feature_level=energy_feature_level, pitch_quantization=pitch_quantization, energy_quantization=energy_quantization, dropout_rate=dropout_rate, eps=eps)
# %%

# %%

# %%

# %%
