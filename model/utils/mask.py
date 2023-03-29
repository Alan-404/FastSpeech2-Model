import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask



def generate_padding_mask(tensor: torch.Tensor)-> torch.Tensor:
    return torch.Tensor(tensor == 0).type(torch.int64)[:, np.newaxis, np.newaxis, :]

def __generate_look_ahead_mask(length: int) -> torch.Tensor:
    return torch.triu(torch.ones((length, length)), diagonal=1)

def generate_look_ahead_mask(tensor: torch.Tensor):
    padding_mask = generate_padding_mask(tensor).to(device)

    look_ahead_mask = __generate_look_ahead_mask(tensor.size(1)).to(device)

    look_ahead_mask = torch.maximum(look_ahead_mask, padding_mask).to(device)

    return look_ahead_mask.to(device)