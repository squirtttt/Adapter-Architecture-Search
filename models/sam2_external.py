import os
import sys
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters.factory import build_adapter
from models import register
from .iou_loss import IOU
from .mmseg.models.sam import MaskDecoder, TwoWayTransformer
from .sam_v2 import BBCEWithLogitLoss, PositionEmbeddingRandom, _iou_loss

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_LOCAL_SAM2_ROOT = os.path.join(_REPO_ROOT, "third_party", "sam2")
if os.path.isdir(os.path.join(_LOCAL_SAM2_ROOT, "sam2")) and _LOCAL_SAM2_ROOT not in sys.path:
    sys.path.insert(0, _LOCAL_SAM2_ROOT)

try:
    from sam2.build_sam import build_sam2
except Exception:  # pragma: no cover - optional external dependency
    build_sam2 = None


def _sam2_missing_error() -> RuntimeError:
    return RuntimeError(
        "The optional SAM2 backend is not available in this environment. "
        "Create a SAM2-capable environment, install facebookresearch/sam2, "
        "and set model.args.sam2.config plus model.args.sam2.checkpoint in the yaml."
    )


def _call_build_sam2(model_cfg: str, checkpoint: Optional[str], device: torch.device):
    if build_sam2 is None:
        raise _sam2_missing_error()

    call_variants = (
        partial(build_sam2, model_cfg, checkpoint, device=device, mode="train"),
        partial(build_sam2, model_cfg, checkpoint, device=device),
        partial(build_sam2, model_cfg, checkpoint),
    )
    last_error = None
    for build in call_variants:
        try:
            return build()
        except TypeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not build SAM2 with the installed sam2 package API: {last_error}")


def _select_sam2_feature(encoder_output: Any) -> torch.Tensor:
    if torch.is_tensor(encoder_output):
        return encoder_output
    if isinstance(encoder_output, dict):
        if torch.is_tensor(encoder_output.get("vision_features")):
            return encoder_output["vision_features"]
        backbone_fpn = encoder_output.get("backbone_fpn")
        if isinstance(backbone_fpn, (list, tuple)) and backbone_fpn:
            for feature in reversed(backbone_fpn):
                if torch.is_tensor(feature):
                    return feature
    if isinstance(encoder_output, (list, tuple)):
        for feature in reversed(encoder_output):
            if torch.is_tensor(feature):
                return feature
    raise RuntimeError("Could not find a tensor feature map in SAM2 image_encoder output.")


class ResidualFeatureAdapterLayer(nn.Module):
    """Adapter layer for SAM2 image-encoder feature maps.

    The shared adapter primitives in this repo operate on NHWC token grids, so
    SAM2 NCHW feature maps are transposed around the adapter call.
    """

    def __init__(
        self,
        adapter_config: Dict,
        embed_dim: int,
        image_size: int,
        patch_size: int,
        gamma: float,
    ) -> None:
        super().__init__()
        primitive_type = adapter_config.get("primitive_type", "identity")
        self.adapter = build_adapter(
            primitive_type=primitive_type,
            dim=adapter_config.get("dim"),
            activation=adapter_config.get("activation"),
            rank=adapter_config.get("rank"),
            gate=adapter_config.get("gate"),
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            freq_mode=adapter_config.get("freq_mode"),
            attention_type=adapter_config.get("attention_type"),
            edge_mode=adapter_config.get("edge_mode"),
        )
        self.register_buffer("gamma", torch.tensor(float(gamma), dtype=torch.float32))

    def set_gamma(self, gamma: float) -> None:
        self.gamma.fill_(float(gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if float(self.gamma.item()) == 0.0:
            return x
        gamma = self.gamma.to(dtype=x.dtype, device=x.device)
        y = x.permute(0, 2, 3, 1).contiguous()
        y = self.adapter(y)
        y = y.permute(0, 3, 1, 2).contiguous()
        return x + gamma * y


class ResidualTokenAdapterLayer(nn.Module):
    def __init__(
        self,
        adapter_config: Dict,
        embed_dim: int,
        image_size: int,
        patch_size: int,
        gamma: float,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter = build_adapter(
            primitive_type=adapter_config.get("primitive_type", "identity"),
            dim=adapter_config.get("dim"),
            activation=adapter_config.get("activation"),
            rank=adapter_config.get("rank"),
            gate=adapter_config.get("gate"),
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            freq_mode=adapter_config.get("freq_mode"),
            attention_type=adapter_config.get("attention_type"),
            edge_mode=adapter_config.get("edge_mode"),
        )
        self.register_buffer("gamma", torch.tensor(float(gamma), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if float(self.gamma.item()) == 0.0:
            return x
        gamma = self.gamma.to(dtype=x.dtype, device=x.device)
        if x.dim() == 4 and x.shape[-1] == self.embed_dim:
            return x + gamma * self.adapter(x)
        if x.dim() == 4 and x.shape[1] == self.embed_dim:
            y = x.permute(0, 2, 3, 1).contiguous()
            y = self.adapter(y).permute(0, 3, 1, 2).contiguous()
            return x + gamma * y
        raise RuntimeError(f"Cannot apply adapter to SAM2 block output shape={tuple(x.shape)}")


class SAM2BlockWithAdapter(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        adapter_config: Dict,
        embed_dim: int,
        image_size: int,
        patch_size: int,
        gamma: float,
    ) -> None:
        super().__init__()
        self.block = block
        self.extended_adapters = ResidualTokenAdapterLayer(
            adapter_config=adapter_config,
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            gamma=gamma,
        )

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if torch.is_tensor(out):
            return self.extended_adapters(out)
        if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
            return (self.extended_adapters(out[0]), *out[1:])
        return out


def _infer_block_embed_dim(block: nn.Module) -> Optional[int]:
    for attr in ("dim", "embed_dim", "out_dim"):
        value = getattr(block, attr, None)
        if isinstance(value, int):
            return value
    for module in block.modules():
        normalized_shape = getattr(module, "normalized_shape", None)
        if isinstance(normalized_shape, tuple) and normalized_shape and isinstance(normalized_shape[-1], int):
            return normalized_shape[-1]
        if isinstance(module, nn.Linear):
            return module.in_features
    return None


def _try_wrap_sam2_blocks(
    image_encoder: nn.Module,
    adapter_config: Dict,
    image_size: int,
    patch_size: int,
    gamma,
) -> bool:
    trunk = getattr(image_encoder, "trunk", None)
    owner = trunk if trunk is not None and hasattr(trunk, "blocks") else image_encoder
    blocks = getattr(owner, "blocks", None)
    if blocks is None or not isinstance(blocks, (nn.ModuleList, list, tuple)):
        return False
    if len(blocks) != len(gamma):
        return False

    wrapped = []
    for block, gamma_value in zip(blocks, gamma):
        embed_dim = _infer_block_embed_dim(block)
        if embed_dim is None:
            return False
        wrapped.append(
            SAM2BlockWithAdapter(
                block=block,
                adapter_config=adapter_config,
                embed_dim=embed_dim,
                image_size=image_size,
                patch_size=patch_size,
                gamma=gamma_value,
            )
        )
    owner.blocks = nn.ModuleList(wrapped)
    return True


class SAM2ImageEncoderWithAdapters(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        adapter_config: Optional[Dict],
        embed_dim: int,
        image_size: int,
        patch_size: int,
        depth: int,
        insertion: str = "auto",
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.img_size = image_size
        self.embed_dim = embed_dim
        self.depth = depth

        adapter_config = adapter_config or {}
        gamma = adapter_config.get("gamma", [0.0] * depth)
        if len(gamma) != depth:
            raise ValueError(f"adapter gamma length must match depth={depth}, got {len(gamma)}")
        self.insertion = insertion
        self.uses_block_adapters = False
        if insertion in {"auto", "block"}:
            self.uses_block_adapters = _try_wrap_sam2_blocks(
                self.image_encoder,
                adapter_config=adapter_config,
                image_size=image_size,
                patch_size=patch_size,
                gamma=gamma,
            )
            if insertion == "block" and not self.uses_block_adapters:
                raise RuntimeError("Could not find compatible SAM2 image_encoder.trunk.blocks for block adapter insertion.")

        if self.uses_block_adapters:
            self.extended_adapters = nn.ModuleList()
        else:
            self.extended_adapters = nn.ModuleList(
                [
                    ResidualFeatureAdapterLayer(
                        adapter_config=adapter_config,
                        embed_dim=embed_dim,
                        image_size=image_size,
                        patch_size=patch_size,
                        gamma=gamma[i],
                    )
                    for i in range(depth)
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = _select_sam2_feature(self.image_encoder(x))
        if features.dim() != 4:
            raise RuntimeError(f"SAM2 image feature must be NCHW, got shape={tuple(features.shape)}")
        if features.shape[1] != self.embed_dim:
            raise RuntimeError(
                f"SAM2 image feature channel mismatch: expected {self.embed_dim}, got {features.shape[1]}. "
                "Set model.args.sam2.feature_dim to the selected SAM2 image feature channel count."
            )
        if self.uses_block_adapters:
            return features
        for adapter in self.extended_adapters:
            features = adapter(features)
        return features


class _SAM2SegmentationBase(nn.Module):
    def __init__(self, inp_size=None, loss=None, sam2=None, adapter=None):
        super().__init__()
        sam2 = sam2 or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inp_size = inp_size or sam2.get("image_size", 1024)
        self.feature_dim = sam2.get("feature_dim", sam2.get("hidden_dim", 256))
        self.prompt_embed_dim = sam2.get("prompt_embed_dim", self.feature_dim)
        self.patch_size = sam2.get("patch_size", 16)
        self.image_embedding_size = self.inp_size // self.patch_size

        model_cfg = sam2.get("config")
        if not model_cfg:
            raise ValueError("model.args.sam2.config is required for SAM2 models")
        sam2_model = _call_build_sam2(model_cfg, sam2.get("checkpoint"), torch.device("cpu"))
        image_encoder = getattr(sam2_model, "image_encoder", None)
        if image_encoder is None:
            raise RuntimeError("The built SAM2 model does not expose an image_encoder module.")

        adapter_depth = sam2.get("adapter_depth", sam2.get("depth", 12))
        self.image_encoder = SAM2ImageEncoderWithAdapters(
            image_encoder=image_encoder,
            adapter_config=adapter,
            embed_dim=self.feature_dim,
            image_size=self.inp_size,
            patch_size=self.patch_size,
            depth=adapter_depth,
            insertion=sam2.get("adapter_insertion", "auto"),
        )

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=self.prompt_embed_dim, mlp_dim=2048, num_heads=8),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.pe_layer = PositionEmbeddingRandom(self.prompt_embed_dim // 2)
        self.no_mask_embed = nn.Embedding(1, self.prompt_embed_dim)

        self.loss_mode = loss
        if self.loss_mode == "bce":
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        elif self.loss_mode == "bbce":
            self.criterionBCE = BBCEWithLogitLoss()
        elif self.loss_mode == "iou":
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()
        else:
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

    def set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _decode_masks(self, features: torch.Tensor, original_size: int) -> torch.Tensor:
        bs = features.shape[0]
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=features.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, features.shape[-2], features.shape[-1]
        )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.get_dense_pe().to(features.device),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return self.postprocess_masks(low_res_masks, original_size, original_size)

    def forward(self):
        self.features = self.image_encoder(self.input)
        self.pred_mask = self._decode_masks(self.features, self.inp_size)

    def infer(self, input):
        features = self.image_encoder(input)
        return self._decode_masks(features, self.inp_size)

    def postprocess_masks(self, masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...]) -> torch.Tensor:
        masks = F.interpolate(masks, (self.inp_size, self.inp_size), mode="bilinear", align_corners=False)
        masks = masks[..., :input_size, :input_size]
        return F.interpolate(masks, original_size, mode="bilinear", align_corners=False)

    def backward_G(self):
        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask)
        if self.loss_mode == "iou":
            self.loss_G += _iou_loss(self.pred_mask, self.gt_mask)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward_G()
        self.optimizer.step()

    def search_backward(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward_G()


@register("sam2_baseline")
class SAM2Baseline(_SAM2SegmentationBase):
    def __init__(self, inp_size=None, loss=None, sam2=None, adapter=None):
        adapter = adapter or {"primitive_type": "identity", "gamma": [0.0] * (sam2 or {}).get("adapter_depth", 12)}
        super().__init__(inp_size=inp_size, loss=loss, sam2=sam2, adapter=adapter)


@register("sam2_adapter")
class SAM2Adapter(_SAM2SegmentationBase):
    pass


@register("sam2_adapter_search")
class SAM2AdapterSearch(_SAM2SegmentationBase):
    pass
