import torch
import torch.nn as nn

from .common import get_activation


class ChannelAttentionAdapter(nn.Module):
    """Channel recalibration for selective feature-wise domain modulation."""

    def __init__(self, embed_dim: int, dim: int, activation: str = "gelu", attention_type: str = "se", reduction: int = 4) -> None:
        super().__init__()
        self.attention_type = attention_type
        hidden = max(1, dim // reduction)
        self.down = nn.Linear(embed_dim, dim)
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden),
            get_activation(activation),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )
        self.eca = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.up = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        pooled = y.mean(dim=(1, 2))
        if self.attention_type == "eca":
            scale = torch.sigmoid(self.eca(pooled.unsqueeze(1))).squeeze(1)
        else:
            scale = self.attn(pooled)
        scale = scale.view(y.shape[0], 1, 1, y.shape[-1])
        return self.up(y * scale)
