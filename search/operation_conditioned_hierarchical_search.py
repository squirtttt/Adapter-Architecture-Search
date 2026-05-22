import copy
import gc
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import models
import models.repeated_adapter_sam
import models.sam2_external
import utils


def count_adapter_params(module):
    return sum(p.numel() for name, p in module.named_parameters() if "extended_adapters" in name)


def estimate_adapter_macs(decoded, embed_dim=768, image_size=1024, patch_size=16):
    op = decoded["operation"]
    cfg = decoded["config"]
    gamma = decoded["gamma"].detach().cpu()
    h = image_size // patch_size
    w = image_size // patch_size
    tokens = h * w

    dim = cfg.get("dim")
    rank = cfg.get("rank")
    per_layer = 0
    if op == "identity":
        per_layer = 0
    elif op == "mlp":
        per_layer = tokens * (embed_dim * dim + dim * embed_dim)
    elif op == "gated_mlp":
        per_layer = tokens * (embed_dim * (2 * dim) + dim * embed_dim)
    elif op == "dwconv":
        per_layer = tokens * (embed_dim * dim + dim * embed_dim + dim * 3 * 3)
    elif op == "low_rank":
        per_layer = tokens * (embed_dim * rank + rank * embed_dim)
    elif op == "frequency":
        per_layer = tokens * (embed_dim * dim + dim * embed_dim + dim * dim + dim * dim + (2 * dim) * dim)
    elif op == "channel_attention":
        hidden = max(1, dim // 4)
        per_layer = tokens * (embed_dim * dim + dim * embed_dim) + dim * hidden + hidden * dim
    elif op == "edge_aware":
        per_layer = tokens * (embed_dim * dim + dim * embed_dim + dim * dim + dim * 3 * 3)

    actual_layers = int((gamma > 0).sum().item())
    effective_layers = float(gamma.sum().item())
    actual_macs = int(per_layer * actual_layers)
    effective_macs = float(per_layer * effective_layers)
    return {
        "adapter_macs_per_layer": int(per_layer),
        "adapter_actual_macs": actual_macs,
        "adapter_actual_gmacs": actual_macs / 1e9,
        "adapter_effective_macs": effective_macs,
        "adapter_effective_gmacs": effective_macs / 1e9,
        "adapter_active_layers": actual_layers,
        "adapter_effective_layers": effective_layers,
    }


DEFAULT_SEARCH_SPACE = {
    "identity": {
        "dim": [],
        "activation": [],
        "gate": [],
        "rank": [],
        "freq_mode": [],
        "attention_type": [],
        "edge_mode": [],
    },
    "mlp": {"dim": [16, 32, 64, 128], "activation": ["gelu", "relu", "silu"]},
    "gated_mlp": {"dim": [16, 32, 64, 128], "gate": ["geglu", "swiglu"]},
    "dwconv": {"dim": [16, 32, 64, 128], "activation": ["gelu", "relu", "silu"]},
    "low_rank": {"rank": [4, 8, 16]},
    "frequency": {
        "dim": [16, 32, 64],
        "activation": ["gelu", "silu"],
        "freq_mode": ["avg_highpass", "laplacian", "token_smoothing"],
    },
    "channel_attention": {
        "dim": [16, 32, 64],
        "activation": ["gelu", "silu"],
        "attention_type": ["se", "eca"],
    },
    "edge_aware": {
        "dim": [16, 32, 64],
        "activation": ["gelu", "silu"],
        "edge_mode": ["sobel", "laplacian", "dw_gradient"],
    },
}


def freeze_sam_backbone_enable_adapter_training(model: nn.Module) -> None:
    for _, param in model.named_parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if "extended_adapters" in name:
            param.requires_grad_(True)


def set_adapter_config(model_spec: Dict, adapter_kwargs: Dict) -> None:
    args = model_spec.setdefault("args", {})
    if "encoder_mode" in args:
        args["encoder_mode"]["extended_adapter"] = adapter_kwargs
    else:
        args["adapter"] = adapter_kwargs


def get_adapter_efficiency_shape(config: Dict):
    args = config["model"]["args"]
    if "encoder_mode" in args:
        encoder_mode = args["encoder_mode"]
        return (
            encoder_mode.get("embed_dim", 768),
            args.get("inp_size", encoder_mode.get("img_size", 1024)),
            encoder_mode.get("patch_size", 16),
        )
    sam2_cfg = args.get("sam2", {})
    return (
        sam2_cfg.get("feature_dim", sam2_cfg.get("hidden_dim", 256)),
        args.get("inp_size", sam2_cfg.get("image_size", 1024)),
        sam2_cfg.get("patch_size", 16),
    )


def update_zico_stats(model: nn.Module, stats: dict):
    for name, mod in model.named_modules():
        if "extended_adapters" not in name:
            continue
        if isinstance(mod, (nn.Linear, nn.Conv2d)) and mod.weight.grad is not None:
            grad = mod.weight.grad.detach().cpu().reshape(-1).numpy().astype(np.float64, copy=False)
            if name not in stats:
                stats[name] = {
                    "count": 1,
                    "mean": grad.copy(),
                    "m2": np.zeros_like(grad),
                    "sum_abs": np.abs(grad),
                }
                continue

            item = stats[name]
            item["count"] += 1
            delta = grad - item["mean"]
            item["mean"] += delta / item["count"]
            item["m2"] += delta * (grad - item["mean"])
            item["sum_abs"] += np.abs(grad)
    return stats


def calculate_zico(stats: dict):
    if len(stats) == 0:
        return 0.0, {}
    score_sum = 0.0
    score_dict = {}
    for modname, item in stats.items():
        if item["count"] < 2:
            continue
        nsr_std = np.sqrt(item["m2"] / item["count"])
        nonzero_idx = np.nonzero(nsr_std)[0]
        if len(nonzero_idx) == 0:
            continue
        nsr_mean_abs = item["sum_abs"] / item["count"]
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        if tmpsum > 0:
            value = np.log(1 + tmpsum / item["mean"].shape[0])
            score_sum += value
            score_dict[modname] = value
    if len(score_dict) == 0:
        return 0.0, {}
    return score_sum / len(score_dict), score_dict


def compute_adapter_zico(model, train_loader, device, local_rank, max_batches=None):
    if not any(p.requires_grad for n, p in model.named_parameters() if "extended_adapters" in n):
        return 0.0, 0.0, {}

    zico_stats = {}
    model.train()
    loss_sum = 0.0
    loss_count = 0
    total_batches = len(train_loader) if max_batches is None else min(len(train_loader), max_batches)
    pbar = tqdm(total=total_batches, leave=False, desc="zico") if local_rank == 0 else None
    for i, batch in enumerate(train_loader):
        if max_batches is not None and i >= max_batches:
            break
        for k, v in batch.items():
            batch[k] = v.to(device)
        model.zero_grad()
        model.set_input(batch["inp"], batch["gt"])
        model.search_backward()
        gathered_loss = [torch.zeros_like(model.loss_G.detach()) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_loss, model.loss_G.detach())
        loss_sum += sum(loss.item() for loss in gathered_loss)
        loss_count += len(gathered_loss)
        zico_stats = update_zico_stats(model, zico_stats)
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()
    zico_score, score_dict = calculate_zico(zico_stats)
    return loss_sum / loss_count if loss_count else 0.0, zico_score, score_dict


def compute_adapter_naswot(
    model,
    train_loader,
    device,
    local_rank,
    max_batches=None,
    max_patterns=64,
    max_features=8192,
):
    """Adapter-only NASWOT proxy for v2 repeated adapters.

    Hooks collect binarized activation patterns from Linear/Conv2d modules
    inside `extended_adapters`, then compute the NASWOT log determinant.
    """
    patterns = []

    def hook_fn(_module, _inp, out):
        if len(patterns) >= max_patterns:
            return
        if isinstance(out, tuple):
            out = out[0]
        out_flat = out.detach().reshape(out.shape[0], -1)
        if out_flat.shape[1] > max_features:
            stride = max(1, out_flat.shape[1] // max_features)
            out_flat = out_flat[:, ::stride][:, :max_features]
        elif out_flat.shape[1] < max_features:
            pad = max_features - out_flat.shape[1]
            out_flat = torch.nn.functional.pad(out_flat, (0, pad))
        binary = (out_flat > 0).float().cpu()
        remaining = max_patterns - len(patterns)
        patterns.extend(binary[:remaining].unbind(0))

    hooks = []
    for name, module in model.named_modules():
        if "extended_adapters" in name and isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(hook_fn))

    if len(hooks) == 0:
        return 0.0, 0.0, {}

    model.train()
    loss_sum = 0.0
    loss_count = 0
    total_batches = len(train_loader) if max_batches is None else min(len(train_loader), max_batches)
    pbar = tqdm(total=total_batches, leave=False, desc="naswot") if local_rank == 0 else None
    try:
        with torch.inference_mode():
            for i, batch in enumerate(train_loader):
                if max_batches is not None and i >= max_batches:
                    break
                if len(patterns) >= max_patterns:
                    break
                for k, v in batch.items():
                    batch[k] = v.to(device)
                model.set_input(batch["inp"], batch["gt"])
                model.forward()
                if hasattr(model, "criterionBCE") and hasattr(model, "pred_mask"):
                    loss = model.criterionBCE(model.pred_mask, model.gt_mask)
                    gathered_loss = [torch.zeros_like(loss.detach()) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_loss, loss.detach())
                    loss_sum += sum(item.item() for item in gathered_loss)
                    loss_count += len(gathered_loss)
                if pbar is not None:
                    pbar.update(1)
    finally:
        for hook in hooks:
            hook.remove()
        if pbar is not None:
            pbar.close()

    if len(patterns) == 0:
        return loss_sum / loss_count if loss_count else 0.0, -1000.0, {}

    all_patterns = torch.stack(patterns, dim=0)

    x = all_patterns.float()
    k_matrix = (x @ x.t()) + ((1.0 - x) @ (1.0 - x.t()))
    _, logdet = np.linalg.slogdet(k_matrix.numpy())
    return loss_sum / loss_count if loss_count else 0.0, float(logdet), {}


class OperationConditionedHierarchicalSearchController:
    """Operation-conditioned hierarchical perturbation search.

    Unlike a global alpha_dim/alpha_act design, this keeps micro-config logits
    under the operation that owns them. A low-rank sample perturbs and updates
    only low_rank/rank; it never touches MLP dim or DWConv activation logits.
    """

    def __init__(self, config: Dict, train_loader, args, log_fn, device, local_rank: int):
        self.config = config
        self.train_loader = train_loader
        self.args = args
        self.log = log_fn
        self.device = device
        self.local_rank = local_rank

        search_cfg = config["search"]
        self.search_strategy = search_cfg.get("search_strategy", "hierarchical_perturbation")
        self.search_proxy = search_cfg.get("search_proxy", "zico").lower()
        if self.search_proxy not in {"zico", "naswot"}:
            raise ValueError(f"Unsupported search_proxy: {self.search_proxy}")
        self.search_mode = search_cfg.get("search_mode", "continuous_residual")
        if self.search_mode == "continuous":
            self.search_mode = "continuous_residual"
        if self.search_mode == "hard":
            self.search_mode = "hard_insertion"

        self.search_space = copy.deepcopy(search_cfg.get("operation_search_space", DEFAULT_SEARCH_SPACE))
        self.identity_as_baseline = search_cfg.get("identity_as_baseline", True)
        self.identity_baseline = None
        if self.identity_as_baseline and "identity" in self.search_space:
            # Identity has no trainable adapter parameters, so adapter-only ZiCo is
            # always zero and it wastes perturbation samples. Keep it as a
            # no-adapter baseline instead of a search operation.
            self.identity_search_space = self.search_space.pop("identity")
        else:
            self.identity_search_space = None
        if self.search_strategy in {"original", "original_zaas"}:
            self.search_strategy = "original_zaas"
            self.search_space = {"mlp": copy.deepcopy(DEFAULT_SEARCH_SPACE["mlp"])}
            self.search_mode = "hard_insertion"

        self.operations = list(self.search_space.keys())
        if not self.operations:
            raise ValueError("operation search space is empty after removing identity baseline")
        self.K = search_cfg.get("K", search_cfg.get("perturbation_number", search_cfg.get("sample_size", 5)))
        self.N = search_cfg.get("N", search_cfg.get("iterations", search_cfg.get("iteration", 20)))
        self.max_proxy_batches = search_cfg.get("max_proxy_batches", search_cfg.get("max_zico_batches"))
        self.max_zico_batches = self.max_proxy_batches
        self.sigma_op = search_cfg.get("sigma_op", 0.1)
        self.sigma_config = search_cfg.get("sigma_config", 0.1)
        self.sigma_layer = search_cfg.get("sigma_layer", 0.1)
        self.eta_op = search_cfg.get("eta_op", 1.0)
        self.eta_config = search_cfg.get("eta_config", 1.0)
        self.eta_layer = search_cfg.get("eta_layer", 1.0)
        self.tau = search_cfg.get("tau", search_cfg.get("threshold", 0.5))
        self.lambda_gamma = search_cfg.get("lambda_gamma", 0.01)
        self.lambda_param = search_cfg.get("lambda_param", 0.0)
        self.softmax_temperature = search_cfg.get("softmax_temperature", search_cfg.get("temperature", 1.0))

        self.alpha_op = torch.zeros(len(self.operations), device=device)
        self.alpha_config = {}
        for op, fields in self.search_space.items():
            self.alpha_config[op] = {}
            for field, values in fields.items():
                if values:
                    self.alpha_config[op][field] = torch.zeros(len(values), device=device)
        self.alpha_layer = torch.tensor(search_cfg.get("alpha_layer", [0.0] * 12), dtype=torch.float32, device=device)

        self.best_score = float("-inf")
        self.best_proxy_score = 0.0
        self.best_zico_score = None
        self.best_naswot_score = None
        self.best_decoded = None
        self.best_adapter_params = 0
        self.search_start_time = None
        self.total_candidate_evals = 0
        self.total_zico_batches = 0
        self.total_candidate_seconds = 0.0
        self.max_peak_memory_mb = 0.0

    def _empty_eps(self):
        return {
            "op": torch.zeros_like(self.alpha_op),
            "config": {
                op: {field: torch.zeros_like(logits) for field, logits in fields.items()}
                for op, fields in self.alpha_config.items()
            },
            "layer": torch.zeros_like(self.alpha_layer),
        }

    def sample_perturbations(self):
        eps = self._empty_eps()
        if self.search_strategy != "random_candidate":
            eps["op"] = torch.randn_like(self.alpha_op) * self.sigma_op
        eps["layer"] = torch.randn_like(self.alpha_layer) * self.sigma_layer

        if self.search_strategy in {"hierarchical_perturbation", "original_zaas"}:
            op = self._decode_op(eps["op"])
            for field, logits in self.alpha_config[op].items():
                eps["config"][op][field] = torch.randn_like(logits) * self.sigma_config
        return eps

    def broadcast_perturbations(self, eps):
        dist.broadcast(eps["op"], src=0)
        dist.broadcast(eps["layer"], src=0)
        for fields in eps["config"].values():
            for value in fields.values():
                dist.broadcast(value, src=0)
        return eps

    def _decode_op(self, eps_op: Optional[torch.Tensor] = None):
        eps_op = torch.zeros_like(self.alpha_op) if eps_op is None else eps_op
        idx = int(torch.argmax(self.alpha_op + eps_op).item())
        return self.operations[idx]

    def _decode_field(self, op: str, field: str, eps_field: Optional[torch.Tensor] = None):
        values = self.search_space[op].get(field, [])
        if not values:
            return None
        logits = self.alpha_config[op][field]
        eps_field = torch.zeros_like(logits) if eps_field is None else eps_field
        idx = int(torch.argmax(logits + eps_field).item())
        return values[idx]

    def _random_decoded_config(self, op: str):
        decoded_config = {}
        for field, values in self.search_space[op].items():
            if not values:
                continue
            idx = torch.randint(len(values), (1,), device=self.device, dtype=torch.long)
            dist.broadcast(idx, src=0)
            decoded_config[field] = values[int(idx.item())]
        return decoded_config

    def decode_candidate(self, eps=None, final: bool = False):
        eps = self._empty_eps() if eps is None else eps
        if self.search_strategy == "random_candidate" and not final:
            op_idx = torch.randint(len(self.operations), (1,), device=self.device, dtype=torch.long)
            dist.broadcast(op_idx, src=0)
            op = self.operations[int(op_idx.item())]
            decoded_config = self._random_decoded_config(op)
        else:
            op = self._decode_op(eps["op"])
            decoded_config = {}
            for field in self.alpha_config[op]:
                decoded_config[field] = self._decode_field(op, field, eps["config"][op].get(field))

        layer_logits = self.alpha_layer + eps["layer"]
        if self.search_mode == "hard_insertion":
            mask = (layer_logits > self.tau).float()
            gamma = mask
        else:
            gamma = torch.sigmoid(layer_logits)
            mask = None

        return {
            "operation": op,
            "config": decoded_config,
            "gamma": gamma.detach().clone(),
            "mask": None if mask is None else mask.detach().clone(),
        }

    def _adapter_kwargs(self, decoded):
        cfg = decoded["config"]
        return {
            "primitive_type": decoded["operation"],
            "dim": cfg.get("dim"),
            "activation": cfg.get("activation"),
            "gate": cfg.get("gate"),
            "rank": cfg.get("rank"),
            "freq_mode": cfg.get("freq_mode"),
            "attention_type": cfg.get("attention_type"),
            "edge_mode": cfg.get("edge_mode"),
            "search_mode": self.search_mode,
            "gamma": [float(v) for v in decoded["gamma"].detach().cpu().tolist()],
        }

    def build_candidate_model(self, decoded):
        cfg = copy.deepcopy(self.config)
        cfg["model"]["name"] = cfg["model"].get("search_model_name", cfg["model"]["name"])
        set_adapter_config(cfg["model"], self._adapter_kwargs(decoded))
        model = models.make(cfg["model"]).cuda()
        optimizer = utils.make_optimizer(model.parameters(), cfg["optimizer"])
        lr_scheduler = CosineAnnealingLR(optimizer, cfg.get("epoch_max"), eta_min=cfg.get("lr_min"))
        model.optimizer = optimizer
        return model, optimizer, lr_scheduler, cfg

    def load_checkpoint_and_freeze(self, model):
        if self.config.get("sam_checkpoint"):
            checkpoint = torch.load(self.config["sam_checkpoint"], map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
        freeze_sam_backbone_enable_adapter_training(model)
        return model

    def compute_score(self, decoded):
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        model, _, _, _ = self.build_candidate_model(decoded)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        ).module
        model = self.load_checkpoint_and_freeze(model)

        if self.search_proxy == "naswot":
            proxy_loss, proxy_score, _ = compute_adapter_naswot(
                model,
                self.train_loader,
                self.device,
                self.local_rank,
                max_batches=self.max_proxy_batches,
            )
        else:
            proxy_loss, proxy_score, _ = compute_adapter_zico(
                model,
                self.train_loader,
                self.device,
                self.local_rank,
                max_batches=self.max_proxy_batches,
            )
        adapter_params = count_adapter_params(model)
        gamma_or_mask = decoded["gamma"] if self.search_mode != "hard_insertion" else decoded["mask"]
        if gamma_or_mask is None:
            gamma_or_mask = decoded["gamma"]
        penalized_score = (
            float(proxy_score)
            - self.lambda_gamma * float(gamma_or_mask.sum().item())
            - self.lambda_param * adapter_params
        )
        elapsed_seconds = time.time() - start_time
        peak_memory_mb = 0.0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        embed_dim, image_size, patch_size = get_adapter_efficiency_shape(self.config)
        macs_info = estimate_adapter_macs(decoded, embed_dim=embed_dim, image_size=image_size, patch_size=patch_size)
        proxy_batches = len(self.train_loader) if self.max_proxy_batches is None else min(len(self.train_loader), self.max_proxy_batches)
        self.total_candidate_evals += 1
        self.total_zico_batches += proxy_batches
        self.total_candidate_seconds += elapsed_seconds
        self.max_peak_memory_mb = max(self.max_peak_memory_mb, peak_memory_mb)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "proxy": self.search_proxy,
            "proxy_score": float(proxy_score),
            "zico_score": float(proxy_score) if self.search_proxy == "zico" else None,
            "naswot_score": float(proxy_score) if self.search_proxy == "naswot" else None,
            "penalized_score": float(penalized_score),
            "adapter_params": int(adapter_params),
            "proxy_loss": float(proxy_loss),
            "elapsed_seconds": elapsed_seconds,
            "peak_memory_mb": peak_memory_mb,
            "proxy_batches": proxy_batches,
            "zico_batches": proxy_batches,
            **macs_info,
        }

    def compute_identity_baseline(self):
        if not self.identity_as_baseline or self.identity_baseline is not None:
            return self.identity_baseline
        decoded = {
            "operation": "identity",
            "config": {},
            "gamma": torch.zeros_like(self.alpha_layer),
            "mask": torch.zeros_like(self.alpha_layer),
        }
        score_info = self.compute_score(decoded)
        self.identity_baseline = {
            "operation": "identity",
            "config": {},
            "gamma": [0.0 for _ in range(len(self.alpha_layer))],
            "proxy": score_info["proxy"],
            "proxy_score": score_info["proxy_score"],
            "zico_score": score_info["zico_score"],
            "naswot_score": score_info["naswot_score"],
            "penalized_score": score_info["penalized_score"],
            "adapter_params": score_info["adapter_params"],
            "elapsed_seconds": score_info.get("elapsed_seconds"),
            "peak_memory_mb": score_info.get("peak_memory_mb"),
            "zico_batches": score_info.get("zico_batches"),
            "adapter_macs_per_layer": score_info.get("adapter_macs_per_layer"),
            "adapter_actual_macs": score_info.get("adapter_actual_macs"),
            "adapter_actual_gmacs": score_info.get("adapter_actual_gmacs"),
            "adapter_effective_macs": score_info.get("adapter_effective_macs"),
            "adapter_effective_gmacs": score_info.get("adapter_effective_gmacs"),
            "adapter_active_layers": score_info.get("adapter_active_layers"),
            "adapter_effective_layers": score_info.get("adapter_effective_layers"),
        }
        if self.local_rank == 0:
            self.log(
                "[identity-baseline] "
                f"{self.search_proxy}={score_info['proxy_score']:.4f} "
                f"score={score_info['penalized_score']:.4f} "
                f"params={score_info['adapter_params']}"
            )
        return self.identity_baseline

    def update_logits(self, perturbations, decoded_samples, scores):
        weights = torch.softmax((scores - scores.max()) / self.softmax_temperature, dim=0)
        if self.search_strategy == "hierarchical_perturbation":
            self.alpha_op = self.alpha_op + self.eta_op * sum(
                weights[k] * perturbations[k]["op"] for k in range(len(perturbations))
            )
            for k, decoded in enumerate(decoded_samples):
                op = decoded["operation"]
                for field in self.alpha_config[op]:
                    self.alpha_config[op][field] = (
                        self.alpha_config[op][field]
                        + self.eta_config * weights[k] * perturbations[k]["config"][op][field]
                    )
        elif self.search_strategy == "original_zaas":
            for k, decoded in enumerate(decoded_samples):
                for field in self.alpha_config["mlp"]:
                    self.alpha_config["mlp"][field] = (
                        self.alpha_config["mlp"][field]
                        + self.eta_config * weights[k] * perturbations[k]["config"]["mlp"][field]
                    )

        self.alpha_layer = self.alpha_layer + self.eta_layer * sum(
            weights[k] * perturbations[k]["layer"] for k in range(len(perturbations))
        )

    def _update_best(self, decoded, score_info):
        if score_info["penalized_score"] > self.best_score:
            self.best_score = score_info["penalized_score"]
            self.best_proxy_score = score_info["proxy_score"]
            self.best_zico_score = score_info["zico_score"]
            self.best_naswot_score = score_info["naswot_score"]
            self.best_decoded = {
                "operation": decoded["operation"],
                "config": copy.deepcopy(decoded["config"]),
                "gamma": decoded["gamma"].detach().cpu(),
                "mask": None if decoded["mask"] is None else decoded["mask"].detach().cpu(),
            }
            self.best_adapter_params = score_info["adapter_params"]

    def _format_decoded(self, decoded):
        return f"op={decoded['operation']}, config={decoded['config']}"

    def run_search(self):
        self.search_start_time = time.time()
        self.compute_identity_baseline()
        for iteration in range(self.N):
            perturbations = []
            decoded_samples = []
            score_infos = []
            scores = []
            for _ in range(self.K):
                eps = self.sample_perturbations() if self.local_rank == 0 else self._empty_eps()
                eps = self.broadcast_perturbations(eps)
                decoded = self.decode_candidate(eps)
                score_info = self.compute_score(decoded)
                perturbations.append(eps)
                decoded_samples.append(decoded)
                score_infos.append(score_info)
                scores.append(score_info["penalized_score"])
                self._update_best(decoded, score_info)

            score_tensor = torch.tensor(scores, device=self.device, dtype=torch.float32)
            dist.broadcast(score_tensor, src=0)
            self.update_logits(perturbations, decoded_samples, score_tensor)

            if self.local_rank == 0:
                parts = []
                for decoded, info in zip(decoded_samples, score_infos):
                    gamma = [round(float(v), 4) for v in decoded["gamma"].detach().cpu().tolist()]
                    parts.append(
                        f"{self._format_decoded(decoded)} gamma={gamma} "
                        f"{self.search_proxy}={info['proxy_score']:.4f} score={info['penalized_score']:.4f} "
                        f"params={info['adapter_params']} "
                        f"gmacs={info['adapter_actual_gmacs']:.3f} "
                        f"time={info['elapsed_seconds']:.1f}s "
                        f"mem={info['peak_memory_mb']:.0f}MB"
                    )
                self.log(
                    f"[op-cond-search {iteration + 1}/{self.N}] "
                    + " || ".join(parts)
                    + f" || best={self.best_decoded['operation'] if self.best_decoded else None}:"
                    + f"{self.best_decoded['config'] if self.best_decoded else None}, best_score={self.best_score:.4f}"
                )
        return self

    def decode_final_architecture(self):
        return self.decode_candidate(final=True)

    def _serialize_decoded(self, decoded, score_info):
        gamma = decoded["gamma"].detach().cpu()
        mask = decoded["mask"]
        if mask is not None:
            mask = mask.detach().cpu()
        return {
            "operation": decoded["operation"],
            "config": copy.deepcopy(decoded["config"]),
            "gamma": [float(v) for v in gamma.tolist()],
            "binary_mask": None if mask is None else [int(v) for v in mask.tolist()],
            "search_mode": self.search_mode,
            "proxy": score_info["proxy"],
            "proxy_score": score_info["proxy_score"],
            "zico_score": score_info["zico_score"],
            "naswot_score": score_info["naswot_score"],
            "penalized_score": score_info["penalized_score"],
            "adapter_params": score_info["adapter_params"],
        }

    def export_best_config(self, save_path: str):
        final_decoded = self.decode_final_architecture()
        final_score = self.compute_score(final_decoded)

        if self.best_decoded is None:
            best_decoded = final_decoded
            best_score = final_score
        else:
            best_decoded = {
                "operation": self.best_decoded["operation"],
                "config": copy.deepcopy(self.best_decoded["config"]),
                "gamma": self.best_decoded["gamma"],
                "mask": self.best_decoded["mask"],
            }
            best_score = {
                "proxy": self.search_proxy,
                "proxy_score": self.best_proxy_score,
                "zico_score": self.best_zico_score,
                "naswot_score": self.best_naswot_score,
                "penalized_score": self.best_score,
                "adapter_params": self.best_adapter_params,
            }

        best_result = self._serialize_decoded(best_decoded, best_score)
        final_result = self._serialize_decoded(final_decoded, final_score)

        result = {
            **best_result,
            "selection": "best_sampled",
            "best_sampled": best_result,
            "final_decoded": final_result,
            "search_space": self.search_space,
            "identity_baseline": self.identity_baseline,
            "efficiency": {
                "total_search_seconds": time.time() - self.search_start_time if self.search_start_time is not None else None,
                "estimated_gpu_hours": (
                    (time.time() - self.search_start_time) * dist.get_world_size() / 3600
                    if self.search_start_time is not None
                    else None
                ),
                "candidate_evaluations": self.total_candidate_evals,
                "proxy_forward_backward_batches": self.total_zico_batches,
                "zico_forward_backward_batches": self.total_zico_batches,
                "candidate_eval_seconds_sum": self.total_candidate_seconds,
                "max_peak_memory_mb": self.max_peak_memory_mb,
            },
            "hyperparameters": {
                "strategy": self.search_strategy,
                "search_proxy": self.search_proxy,
                "max_proxy_batches": self.max_proxy_batches,
                "K": self.K,
                "N": self.N,
                "sigma_op": self.sigma_op,
                "sigma_config": self.sigma_config,
                "sigma_layer": self.sigma_layer,
                "eta_op": self.eta_op,
                "eta_config": self.eta_config,
                "eta_layer": self.eta_layer,
                "lambda_gamma": self.lambda_gamma,
                "lambda_param": self.lambda_param,
                "max_zico_batches": self.max_zico_batches,
            },
        }

        os.makedirs(save_path, exist_ok=True)
        json_path = os.path.join(save_path, "searched_operation_conditioned_zaas_config.json")
        _, _, _, best_cfg = self.build_candidate_model(best_decoded)
        _, _, _, final_cfg = self.build_candidate_model(final_decoded)
        yaml_path = os.path.join(save_path, "best_arch_hierarchical_v2.yaml")
        best_yaml_path = os.path.join(save_path, "best_sampled_arch_hierarchical_v2.yaml")
        final_yaml_path = os.path.join(save_path, "final_decoded_arch_hierarchical_v2.yaml")

        if self.local_rank == 0:
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
            with open(yaml_path, "w") as f:
                yaml.dump(best_cfg, f, sort_keys=False)
            with open(best_yaml_path, "w") as f:
                yaml.dump(best_cfg, f, sort_keys=False)
            with open(final_yaml_path, "w") as f:
                yaml.dump(final_cfg, f, sort_keys=False)
            self.log(f"Operation-conditioned ZAAS JSON saved at: {json_path}")
            self.log(f"Best sampled trainable config saved at: {yaml_path}")
            self.log(f"Best sampled trainable config also saved at: {best_yaml_path}")
            self.log(f"Final decoded trainable config saved at: {final_yaml_path}")
        return result
