import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm2d
from .image_encoder import Block, PatchEmbed
from adapters.factory import build_adapter as build_adapter_from_factory


def _activation(name: Optional[str]) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu" or name == "swish":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class IdentityAdapter(nn.Module):
    """Skip primitive for ablation and sparse adaptation analysis."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class MLPAdapter(nn.Module):
    """Semantic refinement through a lightweight bottleneck residual MLP."""

    def __init__(self, embed_dim: int, dim: int, activation: str = "gelu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, dim),
            _activation(activation),
            nn.Linear(dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedMLPAdapter(nn.Module):
    """Selective modulation with GEGLU/SwiGLU gating in the bottleneck."""

    def __init__(self, embed_dim: int, dim: int, activation: str = "geglu") -> None:
        super().__init__()
        self.gate_type = (activation or "geglu").lower()
        self.down = nn.Linear(embed_dim, dim * 2)
        self.up = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.down(x).chunk(2, dim=-1)
        if self.gate_type == "swiglu":
            gate = F.silu(gate)
        elif self.gate_type == "geglu":
            gate = F.gelu(gate)
        else:
            raise ValueError(f"Unsupported gated activation: {self.gate_type}")
        return self.up(value * gate)


class DWConvAdapter(nn.Module):
    """Texture enhancement by mixing local 3x3 depthwise context cheaply."""

    def __init__(self, embed_dim: int, dim: int, activation: str = "gelu") -> None:
        super().__init__()
        self.down = nn.Linear(embed_dim, dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = _activation(activation)
        self.up = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        y = y.permute(0, 3, 1, 2).contiguous()
        y = self.dwconv(y)
        y = y.permute(0, 2, 3, 1).contiguous()
        return self.up(self.act(y))


class LowRankAdapter(nn.Module):
    """Lightweight residual correction via a low-rank projection B(Ax)."""

    def __init__(self, embed_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.up = nn.Linear(rank, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class FrequencyAdapter(nn.Module):
    """Frequency-aware adaptation separating smooth context and high-frequency residuals."""

    def __init__(self, embed_dim: int, dim: int, activation: str = "gelu") -> None:
        super().__init__()
        self.down = nn.Linear(embed_dim, dim)
        self.low_proj = nn.Linear(dim, dim)
        self.high_proj = nn.Linear(dim, dim)
        self.fuse = nn.Linear(dim * 2, dim)
        self.act = _activation(activation)
        self.up = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        y2d = y.permute(0, 3, 1, 2).contiguous()
        low = F.avg_pool2d(y2d, kernel_size=3, stride=1, padding=1)
        high = y2d - low
        low = low.permute(0, 2, 3, 1).contiguous()
        high = high.permute(0, 2, 3, 1).contiguous()
        fused = torch.cat([self.low_proj(low), self.high_proj(high)], dim=-1)
        return self.up(self.act(self.fuse(fused)))


class ChannelAttentionAdapter(nn.Module):
    """Channel recalibration for selective feature-wise domain modulation."""

    def __init__(self, embed_dim: int, dim: int, activation: str = "gelu", reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, dim // reduction)
        self.down = nn.Linear(embed_dim, dim)
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden),
            _activation(activation),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )
        self.up = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        pooled = y.mean(dim=(1, 2))
        scale = self.attn(pooled).view(y.shape[0], 1, 1, y.shape[-1])
        return self.up(y * scale)


class EdgeAwareAdapter(nn.Module):
    """Boundary refinement with semantic and Laplacian-style edge residual branches."""

    def __init__(self, embed_dim: int, dim: int, activation: str = "gelu") -> None:
        super().__init__()
        self.down = nn.Linear(embed_dim, dim)
        self.semantic = nn.Sequential(_activation(activation), nn.Linear(dim, dim))
        self.edge_scale = nn.Parameter(torch.ones(dim))
        self.up = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        semantic = self.semantic(y)
        y2d = y.permute(0, 3, 1, 2).contiguous()
        low = F.avg_pool2d(y2d, kernel_size=3, stride=1, padding=1)
        edge = (y2d - low).permute(0, 2, 3, 1).contiguous() * self.edge_scale
        return self.up(semantic + edge)


ADAPTER_REGISTRY = {
    "identity": IdentityAdapter,
    "mlp": MLPAdapter,
    "gated_mlp": GatedMLPAdapter,
    "dwconv": DWConvAdapter,
    "low_rank": LowRankAdapter,
    "frequency": FrequencyAdapter,
    "channel_attention": ChannelAttentionAdapter,
    "edge_aware": EdgeAwareAdapter,
}


def build_adapter(
    primitive_type: str,
    embed_dim: int,
    dim: Optional[int],
    activation: Optional[str],
    rank: Optional[int],
    gate: Optional[str] = None,
    image_size: Optional[int] = None,
    patch_size: Optional[int] = None,
    freq_mode: Optional[str] = None,
    attention_type: Optional[str] = None,
    edge_mode: Optional[str] = None,
) -> nn.Module:
    return build_adapter_from_factory(
        primitive_type=primitive_type,
        dim=dim,
        activation=activation,
        rank=rank,
        gate=gate,
        embed_dim=embed_dim,
        image_size=image_size,
        patch_size=patch_size,
        freq_mode=freq_mode,
        attention_type=attention_type,
        edge_mode=edge_mode,
    )


def _legacy_build_adapter(primitive_type: str, embed_dim: int, dim: Optional[int], activation: Optional[str], rank: Optional[int]) -> nn.Module:
    primitive_type = primitive_type.lower()
    if primitive_type == "identity":
        return IdentityAdapter()
    if primitive_type == "low_rank":
        if rank is None:
            raise ValueError("low_rank adapter requires rank")
        return LowRankAdapter(embed_dim, rank)
    if dim is None:
        raise ValueError(f"{primitive_type} adapter requires bottleneck dim")
    if primitive_type == "gated_mlp":
        return GatedMLPAdapter(embed_dim, dim, activation or "geglu")
    if primitive_type in ADAPTER_REGISTRY:
        return ADAPTER_REGISTRY[primitive_type](embed_dim, dim, activation or "gelu")
    raise ValueError(f"Unsupported adapter primitive: {primitive_type}")


class ResidualAdapterLayer(nn.Module):
    def __init__(
        self,
        primitive_type: str,
        embed_dim: int,
        dim: Optional[int],
        activation: Optional[str],
        rank: Optional[int],
        gamma: float,
        gate: Optional[str] = None,
        image_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        freq_mode: Optional[str] = None,
        attention_type: Optional[str] = None,
        edge_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.adapter = build_adapter(
            primitive_type,
            embed_dim,
            dim,
            activation,
            rank,
            gate,
            image_size,
            patch_size,
            freq_mode,
            attention_type,
            edge_mode,
        )
        self.register_buffer("gamma", torch.tensor(float(gamma), dtype=torch.float32))

    def set_gamma(self, gamma: float) -> None:
        self.gamma.fill_(float(gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.to(dtype=x.dtype, device=x.device)
        if float(self.gamma.item()) == 0.0:
            return x
        return x + gamma * self.adapter(x)


class ImageEncoderViTV2(nn.Module):
    """SAM ViT image encoder with one repeated residual adapter candidate across layers."""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        adapter_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_rel_pos=use_rel_pos,
                    rel_pos_zero_init=rel_pos_zero_init,
                    window_size=window_size if i not in global_attn_indexes else 0,
                    input_size=(img_size // patch_size, img_size // patch_size),
                )
            )

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

        adapter_config = adapter_config or {}
        primitive_type = adapter_config.get("primitive_type", "identity")
        dim = adapter_config.get("dim")
        activation = adapter_config.get("activation")
        gate = adapter_config.get("gate")
        rank = adapter_config.get("rank")
        freq_mode = adapter_config.get("freq_mode")
        attention_type = adapter_config.get("attention_type")
        edge_mode = adapter_config.get("edge_mode")
        gamma = adapter_config.get("gamma", [0.0] * depth)
        if len(gamma) != depth:
            raise ValueError(f"adapter gamma length must match depth={depth}, got {len(gamma)}")

        self.extended_adapters = nn.ModuleList(
            [
                ResidualAdapterLayer(
                    primitive_type,
                    embed_dim,
                    dim,
                    activation,
                    rank,
                    gamma[i],
                    gate,
                    img_size,
                    patch_size,
                    freq_mode,
                    attention_type,
                    edge_mode,
                )
                for i in range(depth)
            ]
        )
        self.num_stages = self.depth
        self.out_indices = tuple(range(self.num_stages))

    def set_adapter_gamma(self, gamma) -> None:
        if len(gamma) != self.depth:
            raise ValueError(f"adapter gamma length must match depth={self.depth}, got {len(gamma)}")
        for layer, value in zip(self.extended_adapters, gamma):
            layer.set_gamma(float(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.extended_adapters[i](x)

        return self.neck(x.permute(0, 3, 1, 2))


def count_adapter_params(module: nn.Module) -> int:
    return sum(p.numel() for name, p in module.named_parameters() if "extended_adapters" in name)
