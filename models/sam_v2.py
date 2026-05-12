import logging
from functools import partial
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .iou_loss import IOU
from .mmseg.models.sam import MaskDecoder, TwoWayTransformer
from .mmseg.models.sam.image_encoder_v2 import ImageEncoderViTV2

logger = logging.getLogger(__name__)


class BBCEWithLogitLoss(nn.Module):
    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1.0 - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)
        return w_neg * nn.BCEWithLogitsLoss(pos_weight=ratio)(pred, gt)


def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    return (1 - (inter / union)).mean()


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)


@register("sam_v2")
class SAMV2(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode["embed_dim"]
        self.image_encoder = ImageEncoderViTV2(
            img_size=inp_size,
            patch_size=encoder_mode["patch_size"],
            in_chans=3,
            embed_dim=encoder_mode["embed_dim"],
            depth=encoder_mode["depth"],
            num_heads=encoder_mode["num_heads"],
            mlp_ratio=encoder_mode["mlp_ratio"],
            out_chans=encoder_mode["out_chans"],
            qkv_bias=encoder_mode["qkv_bias"],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode["use_rel_pos"],
            rel_pos_zero_init=True,
            window_size=encoder_mode["window_size"],
            global_attn_indexes=encoder_mode["global_attn_indexes"],
            adapter_config=encoder_mode.get("extended_adapter", {}),
        )
        self.prompt_embed_dim = encoder_mode["prompt_embed_dim"]
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=self.prompt_embed_dim, mlp_dim=2048, num_heads=8),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.loss_mode = loss
        if self.loss_mode == "bce":
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        elif self.loss_mode == "bbce":
            self.criterionBCE = BBCEWithLogitLoss()
        elif self.loss_mode == "iou":
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode["prompt_embed_dim"] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode["patch_size"]
        self.no_mask_embed = nn.Embedding(1, encoder_mode["prompt_embed_dim"])

    def set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self):
        bs = self.input.shape[0]
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        self.features = self.image_encoder(self.input)
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        self.pred_mask = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)

    def infer(self, input):
        bs = input.shape[0]
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        features = self.image_encoder(input)
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)

    def postprocess_masks(self, masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...]) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
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
