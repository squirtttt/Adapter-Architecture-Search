import torch
import torch.nn as nn


class IdentityAdapter(nn.Module):
    """Prior-preserving skip primitive. It contributes A(x)=0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
