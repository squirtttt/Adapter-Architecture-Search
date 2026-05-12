from typing import Optional

from .channel_attention_adapter import ChannelAttentionAdapter
from .dwconv_adapter import DWConvAdapter
from .edge_aware_adapter import EdgeAwareAdapter
from .frequency_adapter import FrequencyAdapter
from .gated_mlp_adapter import GatedMLPAdapter
from .identity_adapter import IdentityAdapter
from .lowrank_adapter import LowRankAdapter
from .mlp_adapter import MLPAdapter


def build_adapter(
    primitive_type: str,
    dim: Optional[int],
    activation: Optional[str],
    rank: Optional[int],
    gate: Optional[str],
    embed_dim: int,
    image_size: Optional[int] = None,
    patch_size: Optional[int] = None,
    freq_mode: Optional[str] = None,
    attention_type: Optional[str] = None,
    edge_mode: Optional[str] = None,
):
    """Build one lightweight residual adapter from decoded hierarchical logits."""
    primitive_type = primitive_type.lower()
    if primitive_type == "identity":
        return IdentityAdapter()
    if primitive_type == "low_rank":
        if rank is None:
            raise ValueError("low_rank adapter requires rank")
        return LowRankAdapter(embed_dim=embed_dim, rank=rank)
    if dim is None:
        raise ValueError(f"{primitive_type} adapter requires a bottleneck dim")
    if primitive_type == "mlp":
        return MLPAdapter(embed_dim=embed_dim, dim=dim, activation=activation or "gelu")
    if primitive_type == "gated_mlp":
        return GatedMLPAdapter(embed_dim=embed_dim, dim=dim, gate=gate or activation or "geglu")
    if primitive_type == "dwconv":
        return DWConvAdapter(embed_dim=embed_dim, dim=dim, activation=activation or "gelu")
    if primitive_type == "frequency":
        return FrequencyAdapter(embed_dim=embed_dim, dim=dim, activation=activation or "gelu", freq_mode=freq_mode or "avg_highpass")
    if primitive_type == "channel_attention":
        return ChannelAttentionAdapter(embed_dim=embed_dim, dim=dim, activation=activation or "gelu", attention_type=attention_type or "se")
    if primitive_type == "edge_aware":
        return EdgeAwareAdapter(embed_dim=embed_dim, dim=dim, activation=activation or "gelu", edge_mode=edge_mode or "dw_gradient")
    raise ValueError(f"Unsupported adapter primitive: {primitive_type}")
