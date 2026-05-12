import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMLPAdapter(nn.Module):
    """Selective modulation with GEGLU/SwiGLU gating inside the bottleneck."""

    def __init__(self, embed_dim: int, dim: int, gate: str = "geglu") -> None:
        super().__init__()
        self.gate = (gate or "geglu").lower()
        self.down = nn.Linear(embed_dim, dim * 2)
        self.up = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.down(x).chunk(2, dim=-1)
        if self.gate == "geglu":
            gate = F.gelu(gate)
        elif self.gate == "swiglu":
            gate = F.silu(gate)
        else:
            raise ValueError(f"Unsupported gate: {self.gate}")
        return self.up(value * gate)
