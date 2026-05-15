import argparse
import copy
import json
import os
from dataclasses import dataclass, field
from statistics import mean
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import models.sam_v2  # registers sam_v2 without modifying the original model registry file
import utils
from models.mmseg.models.sam.image_encoder_v2 import count_adapter_params
from search.operation_conditioned_hierarchical_search import OperationConditionedHierarchicalSearchController


torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


@dataclass
class AdapterCandidate:
    primitive_type: str
    dim: Optional[int] = None
    activation: Optional[str] = None
    rank: Optional[int] = None
    attention_config: Optional[dict] = None
    residual_scale_mode: str = "continuous"
    params_count: int = 0
    alpha: torch.Tensor = field(default_factory=lambda: torch.zeros(12))
    best_score: float = float("-inf")
    best_zico_score: float = float("-inf")
    best_gamma_or_mask: Optional[torch.Tensor] = None

    def to_encoder_config(self, gamma: torch.Tensor) -> dict:
        return {
            "primitive_type": self.primitive_type,
            "dim": self.dim,
            "activation": self.activation,
            "rank": self.rank,
            "attention_config": self.attention_config,
            "residual_scale_mode": self.residual_scale_mode,
            "gamma": [float(v) for v in gamma.detach().cpu().tolist()],
        }


def make_data_loader(spec, tag=""):
    if spec is None:
        return None
    dataset = datasets.make(spec["dataset"])
    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})
    if local_rank == 0:
        log(f"{tag} dataset: size={len(dataset)}")
        for k, v in dataset[0].items():
            log(f"  {k}: shape={tuple(v.shape)}")
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    return DataLoader(dataset, batch_size=spec["batch_size"], shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)


def make_data_loaders():
    return make_data_loader(config.get("train_dataset"), tag="train"), make_data_loader(config.get("val_dataset"), tag="val")


def freeze_sam_backbone_enable_adapter_training(model: nn.Module) -> None:
    for _, param in model.named_parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if "extended_adapters" in name:
            param.requires_grad_(True)


def getgrad(model: nn.Module, grad_dict: dict, step_iter=0):
    for name, mod in model.named_modules():
        if "extended_adapters" not in name:
            continue
        if isinstance(mod, (nn.Linear, nn.Conv2d)) and mod.weight.grad is not None:
            key = name
            grad = mod.weight.grad.data.cpu().reshape(-1).numpy()
            if step_iter == 0 or key not in grad_dict:
                grad_dict[key] = [grad]
            else:
                grad_dict[key].append(grad)
    return grad_dict


def calculate_zico(grad_dict):
    if len(grad_dict) == 0:
        return float("-inf"), {}
    score_sum = 0.0
    score_dict = {}
    for modname in list(grad_dict.keys()):
        grads = np.array(grad_dict[modname])
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
        return float("-inf"), {}
    return score_sum / len(score_dict), score_dict


def getzico(model, train_loader):
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
        grad_dict = getgrad(model, grad_dict, i)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    score, score_dict = calculate_zico(grad_dict)
    losses = [i.item() for i in loss_list]
    return mean(losses) if losses else 0.0, score, score_dict


def build_candidate_pool(search_cfg) -> List[AdapterCandidate]:
    primitives = search_cfg.get(
        "primitives",
        ["identity", "mlp", "gated_mlp", "dwconv", "low_rank", "frequency", "channel_attention", "edge_aware"],
    )
    dims = search_cfg.get("dims", [16, 32, 64, 128])
    acts = search_cfg.get("activations", ["gelu", "relu", "silu"])
    gates = search_cfg.get("gates", ["geglu", "swiglu"])
    ranks = search_cfg.get("ranks", [4, 8, 16])
    alpha_init = torch.tensor(search_cfg.get("alpha", [0.0] * 12), dtype=torch.float32)

    candidates = []
    for primitive in primitives:
        primitive = primitive.lower()
        if primitive == "identity":
            candidates.append(AdapterCandidate("identity", alpha=alpha_init.clone()))
        elif primitive == "mlp":
            for dim in dims:
                for act in acts:
                    candidates.append(AdapterCandidate("mlp", dim=dim, activation=act, alpha=alpha_init.clone()))
        elif primitive == "gated_mlp":
            for dim in dims:
                for gate in gates:
                    candidates.append(AdapterCandidate("gated_mlp", dim=dim, activation=gate, alpha=alpha_init.clone()))
        elif primitive in {"dwconv", "frequency", "channel_attention", "edge_aware"}:
            for dim in dims:
                for act in acts:
                    candidates.append(AdapterCandidate(primitive, dim=dim, activation=act, alpha=alpha_init.clone()))
        elif primitive == "low_rank":
            for rank in ranks:
                candidates.append(AdapterCandidate("low_rank", rank=rank, alpha=alpha_init.clone()))
        else:
            raise ValueError(f"Unsupported primitive in search space: {primitive}")
    return candidates


def decode_alpha(alpha: torch.Tensor, search_mode: str, threshold: float):
    if search_mode == "hard":
        mask = (alpha > threshold).float()
        return mask, mask
    if search_mode == "continuous":
        gamma = torch.sigmoid(alpha)
        return gamma, None
    raise ValueError(f"search_mode must be 'continuous' or 'hard', got {search_mode}")


def prepare_training(candidate: AdapterCandidate, gamma: torch.Tensor, base_config):
    cfg = copy.deepcopy(base_config)
    cfg["model"]["name"] = "sam_v2"
    cfg["model"]["args"]["encoder_mode"]["extended_adapter"] = candidate.to_encoder_config(gamma)
    model = models.make(cfg["model"]).cuda()
    optimizer = utils.make_optimizer(model.parameters(), cfg["optimizer"])
    lr_scheduler = CosineAnnealingLR(optimizer, cfg.get("epoch_max"), eta_min=cfg.get("lr_min"))
    return model, optimizer, 1, lr_scheduler, cfg


def load_checkpoint_and_freeze(model):
    sam_checkpoint = torch.load(config["sam_checkpoint"], map_location="cpu")
    model.load_state_dict(sam_checkpoint, strict=False)
    freeze_sam_backbone_enable_adapter_training(model)
    return model


class ExtendedZAASSearchController:
    def __init__(self, search_cfg, candidates: List[AdapterCandidate]):
        self.candidates = candidates
        self.search_mode = search_cfg.get("search_mode", "continuous")
        self.perturbation_sigma = search_cfg.get("perturbation_sigma", search_cfg.get("epsilon", 0.1))
        self.perturbation_number = search_cfg.get("perturbation_number", search_cfg.get("sample_size", 5))
        self.iterations = search_cfg.get("iterations", search_cfg.get("iteration", 20))
        self.threshold = search_cfg.get("threshold", 0.5)
        self.lambda_gamma = search_cfg.get("lambda_gamma", 0.01)
        self.lambda_param = search_cfg.get("lambda_param", 0.0)
        self.eta = search_cfg.get("eta", 1.0)
        self.softmax_temperature = search_cfg.get("tau", 1.0)

    def evaluate(self, candidate: AdapterCandidate, alpha_k: torch.Tensor, train_loader):
        gamma_k, mask_k = decode_alpha(alpha_k, self.search_mode, self.threshold)
        model, optimizer, _, _, _ = prepare_training(candidate, gamma_k, config)
        model.optimizer = optimizer
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        ).module
        model = load_checkpoint_and_freeze(model)

        _, zico_score, _ = getzico(model, train_loader)
        adapter_params = count_adapter_params(model)
        candidate.params_count = adapter_params
        sparsity_base = gamma_k if self.search_mode == "continuous" else mask_k
        final_score = zico_score - self.lambda_gamma * float(sparsity_base.sum().item()) - self.lambda_param * adapter_params
        torch.cuda.empty_cache()
        return {
            "score": float(final_score),
            "zico_score": float(zico_score),
            "gamma": gamma_k.detach().clone(),
            "mask": None if mask_k is None else mask_k.detach().clone(),
            "adapter_params": adapter_params,
        }

    def search(self, train_loader):
        for idx, candidate in enumerate(self.candidates):
            alpha = candidate.alpha.to(device)
            if local_rank == 0:
                log(f"[candidate {idx + 1}/{len(self.candidates)}] {candidate}")

            for iteration in range(self.iterations):
                if local_rank == 0:
                    eps_list = [torch.randn_like(alpha) * self.perturbation_sigma for _ in range(self.perturbation_number)]
                else:
                    eps_list = [torch.zeros_like(alpha) for _ in range(self.perturbation_number)]
                for eps in eps_list:
                    dist.broadcast(eps, src=0)

                results = []
                scores = []

                for eps in eps_list:
                    alpha_k = alpha + eps
                    result = self.evaluate(candidate, alpha_k, train_loader)
                    results.append(result)
                    scores.append(result["score"])

                score_tensor = torch.tensor(scores, device=device, dtype=torch.float32)
                dist.broadcast(score_tensor, src=0)
                weights = torch.softmax((score_tensor - score_tensor.max()) / self.softmax_temperature, dim=0)
                delta_alpha = sum(weights[k] * eps_list[k] for k in range(len(eps_list)))
                alpha = alpha + self.eta * delta_alpha

                best_idx = int(torch.argmax(score_tensor).item())
                if results[best_idx]["score"] > candidate.best_score:
                    candidate.best_score = results[best_idx]["score"]
                    candidate.best_zico_score = results[best_idx]["zico_score"]
                    candidate.best_gamma_or_mask = results[best_idx]["gamma" if self.search_mode == "continuous" else "mask"]
                    candidate.params_count = results[best_idx]["adapter_params"]

                if local_rank == 0:
                    gamma_log = [round(float(v), 4) for v in candidate.best_gamma_or_mask.detach().cpu().tolist()]
                    log(
                        f"[candidate {idx + 1} | iter {iteration + 1}/{self.iterations}] "
                        f"current={scores[best_idx]:.4f}, best={candidate.best_score:.4f}, gamma/mask={gamma_log}"
                    )

            candidate.alpha = alpha.detach().cpu()
        return max(self.candidates, key=lambda c: c.best_score)


def export_result(best_candidate: AdapterCandidate, save_path: str, base_config):
    final_values = best_candidate.best_gamma_or_mask.detach().cpu()
    if controller.search_mode == "continuous":
        final_gamma = final_values
        final_mask = None
    else:
        final_mask = final_values.int()
        final_gamma = final_values.float()

    final_model, _, _, _, final_cfg = prepare_training(best_candidate, final_gamma, base_config)
    adapter_params = count_adapter_params(final_model)
    result = {
        "primitive_type": best_candidate.primitive_type,
        "dim": best_candidate.dim,
        "activation": best_candidate.activation,
        "rank": best_candidate.rank,
        "search_mode": controller.search_mode,
        "gamma": [float(v) for v in final_gamma.tolist()],
        "binary_mask": None if final_mask is None else [int(v) for v in final_mask.tolist()],
        "zico_score": float(best_candidate.best_zico_score),
        "final_score": float(best_candidate.best_score),
        "adapter_params": int(adapter_params),
    }
    os.makedirs(save_path, exist_ok=True)
    json_path = os.path.join(save_path, "searched_extended_zaas_config.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    final_cfg["model"]["args"]["encoder_mode"]["extended_adapter"] = best_candidate.to_encoder_config(final_gamma)
    yaml_path = os.path.join(save_path, "best_arch_v2.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(final_cfg, f, sort_keys=False)

    log(f"Extended ZAAS JSON saved at: {json_path}")
    log(f"Trainable v2 config saved at: {yaml_path}")
    return result


def main(config_, save_path, parsed_args):
    global config, log, writer, args
    config = config_
    args = parsed_args
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, _ = make_data_loaders()
    if config.get("data_norm") is None:
        config["data_norm"] = {"inp": {"sub": [0], "div": [1]}, "gt": {"sub": [0], "div": [1]}}

    controller = OperationConditionedHierarchicalSearchController(
        config=config,
        train_loader=train_loader,
        args=args,
        log_fn=log,
        device=device,
        local_rank=local_rank,
    )
    controller.run_search()
    result = controller.export_best_config(save_path)

    if local_rank == 0:
        log("=" * 60)
        log("Operation-Conditioned Hierarchical ZAAS Search Completed")
        log(f"Final operation: {result['operation']}")
        log(f"Final config: {result['config']}")
        log(f"ZiCo: {result['zico_score']:.4f}, penalized score: {result['penalized_score']:.4f}")
        if "final_decoded" in result:
            final_decoded = result["final_decoded"]
            log(
                "Alpha/logit decoded operation: "
                f"{final_decoded['operation']}, config: {final_decoded['config']}"
            )
            log(
                "Alpha/logit decoded ZiCo: "
                f"{final_decoded['zico_score']:.4f}, "
                f"penalized score: {final_decoded['penalized_score']:.4f}"
            )
        log("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/search_demo_v2.yaml")
    parser.add_argument("--name", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print("config loaded")

    save_name = args.name or "_" + args.config.split("/")[-1][:-len(".yaml")]
    if args.tag is not None:
        save_name += "_" + args.tag
    main(config, os.path.join("./save", save_name), args)
