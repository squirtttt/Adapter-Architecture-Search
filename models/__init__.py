from .models import register, make

try:
    from . import sam  # noqa: F401
except ModuleNotFoundError as exc:
    # SAM2-only environments do not need the legacy mmseg/mmcv-backed SAM model.
    # Keep the registry usable so models.sam2_external can register sam2_adapter.
    if exc.name != "mmcv":
        raise

