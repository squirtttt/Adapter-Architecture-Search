import copy
import gc
import json
import os
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
import utils
from models.mmseg.models.sam.image_encoder_v2 import count_adapter_params


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
        self.max_zico_batches = search_cfg.get("max_zico_batches")
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
        self.best_zico_score = 0.0
        self.best_decoded = None
        self.best_adapter_params = 0

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
        cfg["model"]["name"] = "repeated_adapter_sam"
        cfg["model"]["args"]["encoder_mode"]["extended_adapter"] = self._adapter_kwargs(decoded)
        model = models.make(cfg["model"]).cuda()
        optimizer = utils.make_optimizer(model.parameters(), cfg["optimizer"])
        lr_scheduler = CosineAnnealingLR(optimizer, cfg.get("epoch_max"), eta_min=cfg.get("lr_min"))
        model.optimizer = optimizer
        return model, optimizer, lr_scheduler, cfg

    def load_checkpoint_and_freeze(self, model):
        checkpoint = torch.load(self.config["sam_checkpoint"], map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        freeze_sam_backbone_enable_adapter_training(model)
        return model

    def compute_score(self, decoded):
        model, _, _, _ = self.build_candidate_model(decoded)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        ).module
        model = self.load_checkpoint_and_freeze(model)

        _, zico_score, _ = compute_adapter_zico(
            model,
            self.train_loader,
            self.device,
            self.local_rank,
            max_batches=self.max_zico_batches,
        )
        adapter_params = count_adapter_params(model)
        gamma_or_mask = decoded["gamma"] if self.search_mode != "hard_insertion" else decoded["mask"]
        penalized_score = (
            float(zico_score)
            - self.lambda_gamma * float(gamma_or_mask.sum().item())
            - self.lambda_param * adapter_params
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "zico_score": float(zico_score),
            "penalized_score": float(penalized_score),
            "adapter_params": int(adapter_params),
        }

    def compute_identity_baseline(self):
        if not self.identity_as_baseline or self.identity_baseline is not None:
            return self.identity_baseline
        decoded = {
            "operation": "identity",
            "config": {},
            "gamma": torch.zeros_like(self.alpha_layer),
            "mask": None,
        }
        score_info = self.compute_score(decoded)
        self.identity_baseline = {
            "operation": "identity",
            "config": {},
            "gamma": [0.0 for _ in range(len(self.alpha_layer))],
            "zico_score": score_info["zico_score"],
            "penalized_score": score_info["penalized_score"],
            "adapter_params": score_info["adapter_params"],
        }
        if self.local_rank == 0:
            self.log(
                "[identity-baseline] "
                f"ZiCo={score_info['zico_score']:.4f} "
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
            self.best_zico_score = score_info["zico_score"]
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
                        f"ZiCo={info['zico_score']:.4f} score={info['penalized_score']:.4f} "
                        f"params={info['adapter_params']}"
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

    def export_best_config(self, save_path: str):
        final_decoded = self.decode_final_architecture()
        final_score = self.compute_score(final_decoded)
        gamma = final_decoded["gamma"].detach().cpu()
        mask = final_decoded["mask"]
        if mask is not None:
            mask = mask.detach().cpu()

        result = {
            "operation": final_decoded["operation"],
            "config": final_decoded["config"],
            "gamma": [float(v) for v in gamma.tolist()],
            "binary_mask": None if mask is None else [int(v) for v in mask.tolist()],
            "search_mode": self.search_mode,
            "zico_score": final_score["zico_score"],
            "penalized_score": final_score["penalized_score"],
            "adapter_params": final_score["adapter_params"],
            "search_space": self.search_space,
            "identity_baseline": self.identity_baseline,
            "hyperparameters": {
                "strategy": self.search_strategy,
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
        _, _, _, final_cfg = self.build_candidate_model(final_decoded)
        final_cfg["model"]["name"] = "repeated_adapter_sam"
        yaml_path = os.path.join(save_path, "best_arch_hierarchical_v2.yaml")

        if self.local_rank == 0:
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
            with open(yaml_path, "w") as f:
                yaml.dump(final_cfg, f, sort_keys=False)
            self.log(f"Operation-conditioned ZAAS JSON saved at: {json_path}")
            self.log(f"Trainable final config saved at: {yaml_path}")
        return result
