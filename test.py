#%%
import torch
from typing import Callable
from torch import Tensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.utils.length import LengthRegulator
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
#%%

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

# %%
from model.utils.mask import generate_padding_mask, generate_look_ahead_mask
# %%
a = torch.tensor([[1,2,3], [2,3,4], [2,0 ,0]])
# %%
device = torch.device('cuda')
# %%
a = a.to(device)
# %%
a
# %%
padding = generate_padding_mask(a)
# %%
look = generate_look_ahead_mask(a)
# %%
look
# %%
padding
# %%
padding.shape
# %%
look.shape
# %%
a  =get_mask_from_lengths(a)
# %%

# %%
length = LengthRegulator()


# %%
result = length(a, 10, 3)
# %%

# %%

# %%

# %%

# %%

# %%

# %%
