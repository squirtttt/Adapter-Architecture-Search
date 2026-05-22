try:
    from .builder import build_loss
    from .losses import *  # noqa: F401,F403
except ModuleNotFoundError as exc:
    # The SAM/SAM2 adapter path imports modules under models.mmseg.models.sam
    # but does not use mmseg registries/loss builders. Allow SAM2-only envs
    # without mmcv to import the lightweight SAM components.
    if exc.name != "mmcv":
        raise

    def build_loss(*args, **kwargs):
        raise ModuleNotFoundError("mmcv is required for mmseg build_loss, but it is not installed.")

__all__ = ["build_loss"]
