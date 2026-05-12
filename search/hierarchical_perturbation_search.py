import copy
import json
import os
from statistics import mean
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import models
import models.repeated_adapter_sam  # registers repeated_adapter_sam
import utils
from models.mmseg.models.sam.image_encoder_v2 import count_adapter_params


def freeze_sam_backbone_enable_adapter_training(model: nn.Module) -> None:
    for _, param in model.named_parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if "extended_adapters" in name:
            param.requires_grad_(True)


def collect_adapter_grads(model: nn.Module, grad_dict: dict, step_iter: int = 0):
    for name, mod in model.named_modules():
        if "extended_adapters" not in name:
            continue
        if isinstance(mod, (nn.Linear, nn.Conv2d)) and mod.weight.grad is not None:
            grad = mod.weight.grad.data.cpu().reshape(-1).numpy()
            if step_iter == 0 or name not in grad_dict:
                grad_dict[name] = [grad]
            else:
                grad_dict[name].append(grad)
    return grad_dict


def calculate_zico(grad_dict: dict):
    if len(grad_dict) == 0:
        return 0.0, {}
    score_sum = 0.0
    score_dict = {}
    for modname, grads in grad_dict.items():
        grads = np.array(grads)
        nsr_std = np.std(grads, axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        if len(nonzero_idx) == 0:
            continue
        nsr_mean_abs = np.mean(np.abs(grads), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        if tmpsum > 0:
            value = np.log(1 + tmpsum / grads.shape[1])
            score_sum += value
            score_dict[modname] = value
    if len(score_dict) == 0:
        return 0.0, {}
    return score_sum / len(score_dict), score_dict


def compute_adapter_zico(model, train_loader, device, local_rank):
    if not any(p.requires_grad for n, p in model.named_parameters() if "extended_adapters" in n):
        return 0.0, 0.0, {}

    grad_dict = {}
    model.train()
    loss_list = []
    pbar = tqdm(total=len(train_loader), leave=False, desc="zico") if local_rank == 0 else None

    for i, batch in enumerate(train_loader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        model.zero_grad()
        model.set_input(batch["inp"], batch["gt"])
        model.search_backward()
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        grad_dict = collect_adapter_grads(model, grad_dict, i)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    zico_score, score_dict = calculate_zico(grad_dict)
    losses = [i.item() for i in loss_list]
    return mean(losses) if losses else 0.0, zico_score, score_dict


class HierarchicalPerturbationSearchController:
    """Score-driven hierarchical perturbation search.

    Random candidate search samples adapter identities and only perturbs layer
    importance. This controller instead keeps continuous logits for every
    architectural level and updates all of them with the ZiCo-weighted
    perturbation direction: primitive, dimension, activation/gate/rank, and
    layer residual importance are all optimized by the same ZAAS-style signal.
    """

    def __init__(self, config: Dict, train_loader, args, log_fn, device, local_rank: int):
        self.config = config
        self.train_loader = train_loader
        self.args = args
        self.log = log_fn
        self.device = device
        self.local_rank = local_rank

        search_cfg = config["search"]
        self.search_strategy = search_cfg.get("search_strategy", "hierarchical")
        self.search_mode = search_cfg.get("search_mode", "continuous")
        self.primitives = search_cfg.get(
            "primitives",
            ["identity", "mlp", "gated_mlp", "dwconv", "low_rank", "frequency", "channel_attention", "edge_aware"],
        )
        self.dims = search_cfg.get("dims", [16, 32, 64, 128])
        self.activations = search_cfg.get("activations", ["gelu", "relu", "silu"])
        self.gates = search_cfg.get("gates", ["geglu", "swiglu"])
        self.ranks = search_cfg.get("ranks", [4, 8, 16])
        if self.search_strategy == "original":
            self.primitives = ["mlp"]
            self.search_mode = "hard"

        self.K = search_cfg.get("K", search_cfg.get("perturbation_number", search_cfg.get("sample_size", 5)))
        self.N = search_cfg.get("N", search_cfg.get("iterations", search_cfg.get("iteration", 20)))
        self.sigma_op = search_cfg.get("sigma_op", 0.1)
        self.sigma_dim = search_cfg.get("sigma_dim", 0.1)
        self.sigma_act = search_cfg.get("sigma_act", 0.1)
        self.sigma_gate = search_cfg.get("sigma_gate", self.sigma_act)
        self.sigma_rank = search_cfg.get("sigma_rank", self.sigma_act)
        self.sigma_layer = search_cfg.get("sigma_layer", 0.1)
        self.eta_op = search_cfg.get("eta_op", 1.0)
        self.eta_dim = search_cfg.get("eta_dim", 1.0)
        self.eta_act = search_cfg.get("eta_act", 1.0)
        self.eta_gate = search_cfg.get("eta_gate", self.eta_act)
        self.eta_rank = search_cfg.get("eta_rank", self.eta_act)
        self.eta_layer = search_cfg.get("eta_layer", 1.0)
        self.tau = search_cfg.get("tau", search_cfg.get("threshold", 0.5))
        self.lambda_gamma = search_cfg.get("lambda_gamma", 0.01)
        self.lambda_param = search_cfg.get("lambda_param", 0.0)
        self.softmax_temperature = search_cfg.get("softmax_temperature", search_cfg.get("temperature", 1.0))

        self.alpha_op = torch.zeros(len(self.primitives), device=device)
        self.alpha_dim = torch.zeros(len(self.dims), device=device)
        self.alpha_act = torch.zeros(len(self.activations), device=device)
        self.alpha_gate = torch.zeros(len(self.gates), device=device)
        self.alpha_rank = torch.zeros(len(self.ranks), device=device)
        self.alpha_layer = torch.tensor(search_cfg.get("alpha_layer", [0.0] * 12), dtype=torch.float32, device=device)

        self.best_score = float("-inf")
        self.best_zico_score = 0.0
        self.best_candidate = None
        self.best_gamma = None
        self.best_mask = None
        self.best_adapter_params = 0

    def sample_perturbations(self) -> Dict[str, torch.Tensor]:
        if self.search_strategy == "random":
            return {
                "op": torch.zeros_like(self.alpha_op),
                "dim": torch.zeros_like(self.alpha_dim),
                "act": torch.zeros_like(self.alpha_act),
                "gate": torch.zeros_like(self.alpha_gate),
                "rank": torch.zeros_like(self.alpha_rank),
                "layer": torch.randn_like(self.alpha_layer) * self.sigma_layer,
            }
        return {
            "op": torch.randn_like(self.alpha_op) * self.sigma_op,
            "dim": torch.randn_like(self.alpha_dim) * self.sigma_dim,
            "act": torch.randn_like(self.alpha_act) * self.sigma_act,
            "gate": torch.randn_like(self.alpha_gate) * self.sigma_gate,
            "rank": torch.randn_like(self.alpha_rank) * self.sigma_rank,
            "layer": torch.randn_like(self.alpha_layer) * self.sigma_layer,
        }

    def broadcast_perturbations(self, eps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for value in eps.values():
            dist.broadcast(value, src=0)
        return eps

    def _decode_index(self, logits: torch.Tensor, eps: torch.Tensor, values):
        idx = int(torch.argmax(logits + eps).item())
        return values[idx], idx

    def decode_candidate(self, eps: Optional[Dict[str, torch.Tensor]] = None, final: bool = False) -> Dict:
        zero = {
            "op": torch.zeros_like(self.alpha_op),
            "dim": torch.zeros_like(self.alpha_dim),
            "act": torch.zeros_like(self.alpha_act),
            "gate": torch.zeros_like(self.alpha_gate),
            "rank": torch.zeros_like(self.alpha_rank),
            "layer": torch.zeros_like(self.alpha_layer),
        }
        eps = zero if eps is None else eps

        if self.search_strategy == "random" and not final:
            op_idx = torch.randint(len(self.primitives), (1,), device=self.device).item()
            dim_idx = torch.randint(len(self.dims), (1,), device=self.device).item()
            act_idx = torch.randint(len(self.activations), (1,), device=self.device).item()
            gate_idx = torch.randint(len(self.gates), (1,), device=self.device).item()
            rank_idx = torch.randint(len(self.ranks), (1,), device=self.device).item()
            packed = torch.tensor([op_idx, dim_idx, act_idx, gate_idx, rank_idx], device=self.device, dtype=torch.long)
            dist.broadcast(packed, src=0)
            op_idx, dim_idx, act_idx, gate_idx, rank_idx = [int(v) for v in packed.tolist()]
            primitive = self.primitives[op_idx]
            dim = self.dims[dim_idx]
            activation = self.activations[act_idx]
            gate = self.gates[gate_idx]
            rank = self.ranks[rank_idx]
        else:
            primitive, op_idx = self._decode_index(self.alpha_op, eps["op"], self.primitives)
            dim, dim_idx = self._decode_index(self.alpha_dim, eps["dim"], self.dims)
            activation, act_idx = self._decode_index(self.alpha_act, eps["act"], self.activations)
            gate, gate_idx = self._decode_index(self.alpha_gate, eps["gate"], self.gates)
            rank, rank_idx = self._decode_index(self.alpha_rank, eps["rank"], self.ranks)

        if primitive == "identity":
            dim = None
            activation = None
            gate = None
            rank = None
        elif primitive == "low_rank":
            dim = None
            activation = None
            gate = None
        elif primitive == "gated_mlp":
            activation = None
            rank = None
        else:
            gate = None
            rank = None

        layer_logits = self.alpha_layer + eps["layer"]
        if self.search_mode == "hard":
            mask = (layer_logits > self.tau).float()
            gamma = mask
        else:
            gamma = torch.sigmoid(layer_logits)
            mask = None

        return {
            "primitive_type": primitive,
            "dim": dim,
            "activation": activation,
            "gate": gate,
            "rank": rank,
            "gamma": gamma.detach().clone(),
            "mask": None if mask is None else mask.detach().clone(),
        }

    def build_candidate_model(self, decoded: Dict):
        cfg = copy.deepcopy(self.config)
        cfg["model"]["name"] = "repeated_adapter_sam"
        cfg["model"]["args"]["encoder_mode"]["extended_adapter"] = {
            "primitive_type": decoded["primitive_type"],
            "dim": decoded["dim"],
            "activation": decoded["activation"],
            "gate": decoded["gate"],
            "rank": decoded["rank"],
            "search_mode": self.search_mode,
            "gamma": [float(v) for v in decoded["gamma"].detach().cpu().tolist()],
        }
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

    def compute_score(self, decoded: Dict) -> Dict:
        model, _, _, _ = self.build_candidate_model(decoded)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        ).module
        model = self.load_checkpoint_and_freeze(model)

        _, zico_score, _ = compute_adapter_zico(model, self.train_loader, self.device, self.local_rank)
        adapter_params = count_adapter_params(model)
        gamma_or_mask = decoded["gamma"] if self.search_mode == "continuous" else decoded["mask"]
        sparsity_penalty = self.lambda_gamma * float(gamma_or_mask.sum().item())
        param_penalty = self.lambda_param * adapter_params
        penalized_score = float(zico_score) - sparsity_penalty - param_penalty
        torch.cuda.empty_cache()
        return {
            "zico_score": float(zico_score),
            "penalized_score": penalized_score,
            "adapter_params": int(adapter_params),
        }

    def update_logits(self, perturbations, scores: torch.Tensor) -> None:
        weights = torch.softmax((scores - scores.max()) / self.softmax_temperature, dim=0)

        def weighted_delta(key):
            return sum(weights[k] * perturbations[k][key] for k in range(len(perturbations)))

        if self.search_strategy != "random":
            self.alpha_op = self.alpha_op + self.eta_op * weighted_delta("op")
            if self.search_strategy != "original":
                self.alpha_gate = self.alpha_gate + self.eta_gate * weighted_delta("gate")
                self.alpha_rank = self.alpha_rank + self.eta_rank * weighted_delta("rank")
            self.alpha_dim = self.alpha_dim + self.eta_dim * weighted_delta("dim")
            self.alpha_act = self.alpha_act + self.eta_act * weighted_delta("act")
        self.alpha_layer = self.alpha_layer + self.eta_layer * weighted_delta("layer")

    def _format_candidate(self, decoded: Dict) -> str:
        return (
            f"op={decoded['primitive_type']}, dim={decoded['dim']}, act={decoded['activation']}, "
            f"gate={decoded['gate']}, rank={decoded['rank']}"
        )

    def _update_best(self, decoded: Dict, score_info: Dict) -> None:
        if score_info["penalized_score"] > self.best_score:
            self.best_score = score_info["penalized_score"]
            self.best_zico_score = score_info["zico_score"]
            self.best_candidate = {k: decoded[k] for k in ["primitive_type", "dim", "activation", "gate", "rank"]}
            self.best_gamma = decoded["gamma"].detach().cpu()
            self.best_mask = None if decoded["mask"] is None else decoded["mask"].detach().cpu()
            self.best_adapter_params = score_info["adapter_params"]

    def run_search(self):
        for iteration in range(self.N):
            perturbations = []
            decoded_samples = []
            sample_infos = []
            scores = []

            for _ in range(self.K):
                eps = self.sample_perturbations() if self.local_rank == 0 else {
                    "op": torch.zeros_like(self.alpha_op),
                    "dim": torch.zeros_like(self.alpha_dim),
                    "act": torch.zeros_like(self.alpha_act),
                    "gate": torch.zeros_like(self.alpha_gate),
                    "rank": torch.zeros_like(self.alpha_rank),
                    "layer": torch.zeros_like(self.alpha_layer),
                }
                eps = self.broadcast_perturbations(eps)
                decoded = self.decode_candidate(eps)
                score_info = self.compute_score(decoded)

                perturbations.append(eps)
                decoded_samples.append(decoded)
                sample_infos.append(score_info)
                scores.append(score_info["penalized_score"])
                self._update_best(decoded, score_info)

            score_tensor = torch.tensor(scores, device=self.device, dtype=torch.float32)
            dist.broadcast(score_tensor, src=0)
            self.update_logits(perturbations, score_tensor)

            if self.local_rank == 0:
                best_k = int(torch.argmax(score_tensor).item())
                selected = decoded_samples[best_k]
                info = sample_infos[best_k]
                gamma_values = [round(float(v), 4) for v in selected["gamma"].detach().cpu().tolist()]
                self.log(
                    f"[hier-search {iteration + 1}/{self.N}] selected {self._format_candidate(selected)} | "
                    f"gamma={gamma_values} | ZiCo={info['zico_score']:.4f} | "
                    f"score={info['penalized_score']:.4f} | params={info['adapter_params']} | "
                    f"best={self.best_candidate}, best_score={self.best_score:.4f}"
                )
        return self

    def decode_final_architecture(self) -> Dict:
        return self.decode_candidate(final=True)

    def export_best_config(self, save_path: str) -> Dict:
        final_decoded = self.decode_final_architecture()
        final_score = self.compute_score(final_decoded)
        gamma = final_decoded["gamma"].detach().cpu()
        mask = final_decoded["mask"]
        if mask is not None:
            mask = mask.detach().cpu()

        result = {
            "primitive_type": final_decoded["primitive_type"],
            "dimension": final_decoded["dim"],
            "activation": final_decoded["activation"],
            "gate": final_decoded["gate"],
            "rank": final_decoded["rank"],
            "search_mode": self.search_mode,
            "gamma": [float(v) for v in gamma.tolist()],
            "binary_mask": None if mask is None else [int(v) for v in mask.tolist()],
            "zico_score": final_score["zico_score"],
            "penalized_score": final_score["penalized_score"],
            "adapter_params": final_score["adapter_params"],
            "search_hyperparameters": {
                "strategy": self.search_strategy,
                "K": self.K,
                "N": self.N,
                "sigma_op": self.sigma_op,
                "sigma_dim": self.sigma_dim,
                "sigma_act": self.sigma_act,
                "sigma_layer": self.sigma_layer,
                "lambda_gamma": self.lambda_gamma,
                "lambda_param": self.lambda_param,
            },
        }

        os.makedirs(save_path, exist_ok=True)
        json_path = os.path.join(save_path, "searched_hierarchical_zaas_config.json")
        if self.local_rank == 0:
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)

        _, _, _, final_cfg = self.build_candidate_model(final_decoded)
        final_cfg["model"]["name"] = "repeated_adapter_sam"
        yaml_path = os.path.join(save_path, "best_arch_hierarchical_v2.yaml")
        if self.local_rank == 0:
            with open(yaml_path, "w") as f:
                yaml.dump(final_cfg, f, sort_keys=False)

        if self.local_rank == 0:
            self.log(f"Hierarchical ZAAS JSON saved at: {json_path}")
            self.log(f"Trainable final config saved at: {yaml_path}")
        return result
