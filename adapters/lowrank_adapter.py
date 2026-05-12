import torch
import torch.nn as nn


class LowRankAdapter(nn.Module):
    """Parameter-efficient residual correction A(x)=B(Ax)."""

    def __init__(self, embed_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.up = nn.Linear(rank, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))
