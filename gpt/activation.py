import torch
from torch import nn


class GELU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        act = 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi))
            *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        return act
