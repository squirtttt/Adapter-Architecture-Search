from models import register
from .sam_v2 import SAMV2


@register("repeated_adapter_sam")
class RepeatedAdapterSAM(SAMV2):
    """SAM-Adapter variant that repeats one searched adapter primitive across all ViT layers.

    The hierarchical search controller decodes a single global adapter identity
    (primitive/dim/activation/gate/rank) and a 12-D layer importance vector.
    This model keeps that original ZAAS/SAM-Adapter philosophy: no layer gets
    a different adapter type; only the residual coefficient changes by layer.
    """

    pass
