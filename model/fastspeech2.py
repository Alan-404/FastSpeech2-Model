import torch
import torch.nn as nn
from torch import Tensor

from model.components.encoder import Encoder
from model.utils.postion import PositionalEncoding
from model.components.adaptor import VarianceAdaptor
from model.components.decoder import Decoder

from typing import Callable


class FastSpeech2Model(nn.Module):
    def __init__(self, 
                phoneme_size: int, 
                n: int, 
                d_model: int, 
                heads: int, 
                hidden_dim: int, 
                n_mels: int,
                n_bins: int, 
                kernel_size: int,
                num_layers: int,
                pitch_feature_level: str, 
                energy_feature_level: str, 
                pitch_quantization: str, 
                energy_quantization: str,
                dropout_rate: float, 
                eps: float, 
                activation: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.phoneme_embedding = nn.Embedding(num_embeddings=phoneme_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding()
        self.encoder = Encoder(n=n, d_model=d_model, heads=heads, hidden_dim=hidden_dim, dropout_rate=dropout_rate, eps=eps, activation=activation)
        self.variance_adaptor = VarianceAdaptor(d_model=d_model,hidden_dim=hidden_dim, n_bins=n_bins, pitch_feature_level=pitch_feature_level, energy_feature_level=energy_feature_level, pitch_quantization=pitch_quantization, energy_quantization=energy_quantization, dropout_rate=dropout_rate, eps=eps)
        self.decoder = Decoder(n=n, d_model=d_model, heads=heads, hidden_dim=hidden_dim, dropout_rate=dropout_rate, eps=eps, n_mel=n_mels, kernel_size=kernel_size, num_layers=num_layers, activation=activation)
    def forward(self, x: Tensor):
        pass
        