import torch
import torch.nn as nn

from .common import get_activation


class DWConvAdapter(nn.Module):
    """Texture enhancement by applying cheap 3x3 depthwise spatial mixing."""

    def __init__(self, embed_dim: int, dim: int, activation: str = "gelu") -> None:
        super().__init__()
        self.down = nn.Linear(embed_dim, dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = get_activation(activation)
        self.up = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        y = y.permute(0, 3, 1, 2).contiguous()
        y = self.dwconv(y)
        y = y.permute(0, 2, 3, 1).contiguous()
        return self.up(self.act(y))
