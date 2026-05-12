import torch
import torch.nn as nn

from .common import get_activation


class MLPAdapter(nn.Module):
    """Semantic refinement through a lightweight down-activation-up bottleneck."""

    def __init__(self, embed_dim: int, dim: int, activation: str = "gelu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, dim),
            get_activation(activation),
            nn.Linear(dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
