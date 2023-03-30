import torch
from torch import Tensor
import torch.nn as nn
from model.utils.predictor import VariancePredictor
from model.utils.length import LengthRegulator
import numpy as np

from model.utils.mask import get_mask_from_lengths

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, n_bins: int, pitch_feature_level: str, energy_feature_level: str, pitch_quantization: str, energy_quantization: str, dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.duration_predictor = VariancePredictor(d_model=d_model, hidden_dim=hidden_dim, dropout_rate=dropout_rate, eps=eps)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(d_model=d_model, hidden_dim=hidden_dim, dropout_rate=dropout_rate, eps=eps)
        self.energy_predictor = VariancePredictor(d_model=d_model, hidden_dim=hidden_dim, dropout_rate=dropout_rate, eps=eps)
        
        pitch = [-2.917079304729967, 11.391254536985784, 207.6309860026605, 46.77559025098988]
        energy = [-1.431044578552246, 8.184337615966797, 37.32621679053821, 26.044180782835863]

        self.pitch_feature_level = pitch_feature_level
        self.energy_feature_level = energy_feature_level

        pitch_min, pitch_max = pitch[:2]
        energy_min, energy_max = energy[:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins-1)),
                requires_grad=False
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )

        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(torch.linspace(np.log(energy_min), np.log(energy_max), n_bins-1)),
                requires_grad=False
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.exp(torch.linspace(energy_min, energy_max, n_bins-1)),
                requires_grad=False
            )
        
        self.pitch_embedding = nn.Embedding(num_embeddings=n_bins, embedding_dim=d_model)
        self.energy_embedding = nn.Embedding(num_embeddings=n_bins, embedding_dim=d_model)

        self.to(device)

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, None)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, None)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        log_duration_prediction = self.duration_predictor(x, None)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        elif self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, e_control
            )
            x = x + energy_embedding
        elif self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)


        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )