import torch.nn as nn


class GELU(nn.Module):
    def forward(self, x):
        return nn.functional.gelu(x)
