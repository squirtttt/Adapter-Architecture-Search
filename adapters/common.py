from typing import Optional

import torch.nn as nn


def get_activation(name: Optional[str]) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")
