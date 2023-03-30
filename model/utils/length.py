import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class LengthRegulator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to(device)

    def LR(self, x: Tensor, duration: int, max_len: int):
        output = list()
        mel_len = list()

        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

            if max_len is not None:
                output = self.pad(output, max_len)
            else:
                output = self.pad(output)

            return output, torch.LongTensor(mel_len)

    def expand(self, batch: Tensor, predicted: Tensor):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))

        out = torch.cat(out, 0)
    
    def pad(self, input_ele, mel_max_length=None):
        if mel_max_length:
            max_len = mel_max_length
        else:
            max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

        out_list = list()
        for i, batch in enumerate(input_ele):
            if len(batch.shape) == 1:
                one_batch_padded = F.pad(
                    batch, (0, max_len - batch.size(0)), "constant", 0.0
                )
            elif len(batch.shape) == 2:
                one_batch_padded = F.pad(
                    batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
                )
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded
    
    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len