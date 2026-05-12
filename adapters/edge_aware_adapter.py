import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_activation


class EdgeAwareAdapter(nn.Module):
    """Boundary refinement using a semantic branch plus high-frequency edge residual."""

    def __init__(self, embed_dim: int, dim: int, activation: str = "gelu", edge_mode: str = "dw_gradient") -> None:
        super().__init__()
        self.edge_mode = edge_mode
        self.down = nn.Linear(embed_dim, dim)
        self.semantic = nn.Sequential(get_activation(activation), nn.Linear(dim, dim))
        self.dw_gradient = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.edge_scale = nn.Parameter(torch.ones(dim))
        self.up = nn.Linear(dim, embed_dim)
        self._init_edge_kernel()

    def _init_edge_kernel(self) -> None:
        if self.edge_mode == "sobel":
            kernel = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        elif self.edge_mode == "laplacian":
            kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        else:
            return
        with torch.no_grad():
            self.dw_gradient.weight.copy_(kernel.view(1, 1, 3, 3).repeat(self.dw_gradient.out_channels, 1, 1, 1))
        self.dw_gradient.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        semantic = self.semantic(y)
        y2d = y.permute(0, 3, 1, 2).contiguous()
        if self.edge_mode in {"sobel", "laplacian", "dw_gradient"}:
            edge2d = self.dw_gradient(y2d)
        else:
            edge2d = y2d - F.avg_pool2d(y2d, kernel_size=3, stride=1, padding=1)
        edge = edge2d.permute(0, 2, 3, 1).contiguous() * self.edge_scale
        return self.up(semantic + edge)
