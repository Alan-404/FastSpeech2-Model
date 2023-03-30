import torch
import torch.nn as nn
from torch import Tensor

from model.components.encoder import Encoder
from model.utils.postion import PositionalEncoding
from model.components.adaptor import VarianceAdaptor
from model.components.decoder import Decoder

from typing import Callable

from model.utils.mask import generate_padding_mask


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
    def forward(self, 
                x: Tensor,
                max_mel_len: int,
                pitch_targets: Tensor = None,
                energy_targets: Tensor = None,
                duration_targets: Tensor = None,
                p_control: float = 1.0,
                e_control: float = 1.0,
                d_control: float = 1.0):
        mask = (x==0)
        x = self.phoneme_embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask)
        (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        ) = self.variance_adaptor(
            x, 
            mask, 
            None, 
            max_mel_len, 
            pitch_targets, 
            energy_targets, 
            duration_targets, 
            p_control, 
            e_control, 
            d_control
        )

        x = self.positional_encoding(x)
        x = self.decoder(x, mel_mask)

        return x



        