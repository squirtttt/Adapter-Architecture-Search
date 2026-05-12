import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_activation


class FrequencyAdapter(nn.Module):
    """Frequency-aware adaptation for texture and high-frequency boundaries."""

    def __init__(self, embed_dim: int, dim: int, activation: str = "gelu", freq_mode: str = "avg_highpass") -> None:
        super().__init__()
        self.freq_mode = freq_mode
        self.down = nn.Linear(embed_dim, dim)
        self.low_proj = nn.Linear(dim, dim)
        self.high_proj = nn.Linear(dim, dim)
        self.fuse = nn.Linear(dim * 2, dim)
        self.act = get_activation(activation)
        self.up = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        y2d = y.permute(0, 3, 1, 2).contiguous()
        if self.freq_mode == "token_smoothing":
            low = F.avg_pool2d(y2d, kernel_size=5, stride=1, padding=2)
        elif self.freq_mode == "laplacian":
            smooth = F.avg_pool2d(y2d, kernel_size=3, stride=1, padding=1)
            low = F.avg_pool2d(smooth, kernel_size=3, stride=1, padding=1)
        else:
            low = F.avg_pool2d(y2d, kernel_size=3, stride=1, padding=1)
        high = y2d - low
        low = low.permute(0, 2, 3, 1).contiguous()
        high = high.permute(0, 2, 3, 1).contiguous()
        y = self.fuse(torch.cat([self.low_proj(low), self.high_proj(high)], dim=-1))
        return self.up(self.act(y))
