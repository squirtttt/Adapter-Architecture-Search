import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.distributed as dist


# Global variables for K matrix accumulation
K_matrix = None
local_rank = None

# Store activation patterns across batches for proper K matrix computation
activation_patterns = []


def network_weight_gaussian_init_adapter_only(net: nn.Module):
    # Adapter 모듈 이름 리스트 (예시)
    adapter_module_names = ['adapter_block', 'nas_op']

    for name, module in net.named_modules():
        # Adapter 모듈인지 확인
        is_adapter_module = any(adapter_name in name for adapter_name in adapter_module_names)

        # SAM 백본 레이어는 건너뛰고, Adapter 모듈에 대해서만 초기화 수행
        if not is_adapter_module:
            continue

        # 이전 초기화 로직은 Adapter 모듈에만 적용
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

def logdet(K):
    s, ld = np.linalg.slogdet(K)
    return ld



def counting_forward_hook(module, _inp, out):
    """
    Forward hook to accumulate activation patterns for NASWOT score.

    This hook is registered on Linear layers within lightweight_mlp.
    We collect binarized activation patterns across all batches,
    then compute K matrix at the end to handle batch_size=1 case.
    """
    global activation_patterns

    try:
        # Use OUTPUT of Linear layer (pre-activation values)
        if isinstance(out, tuple):
            out = out[0]

        # Flatten and binarize: x_i = 1 if value > 0, else 0
        out_flat = out.view(out.size(0), -1)
        x = (out_flat > 0).float()

        # Store activation pattern (will compute K matrix later)
        activation_patterns.append(x.cpu())

    except Exception as err:
        print('---- error on counting_forward_hook, module: ', type(module))
        raise err


def caculate_naswot(max_samples=64):
    """
    Calculate NASWOT score from accumulated activation patterns.

    Args:
        max_samples: Maximum number of samples to use for K matrix (for memory efficiency)
    """
    global activation_patterns

    if len(activation_patterns) == 0:
        return -1000.0

    # Concatenate all activation patterns
    all_patterns = torch.cat(activation_patterns, dim=0)

    # Limit samples for memory efficiency
    if all_patterns.size(0) > max_samples:
        indices = torch.randperm(all_patterns.size(0))[:max_samples]
        all_patterns = all_patterns[indices]

    # Compute K matrix
    x = all_patterns.float()
    K = x @ x.t()
    K2 = (1. - x) @ (1. - x.t())
    K_matrix = (K + K2).numpy()

    # Calculate logdet
    score = logdet(K_matrix)

    # Reset for next computation
    activation_patterns = []

    return score


def getnaswot(model, train_loader, arch_config):
    """
    NASWOT score computation for SAM-Adapter architecture search.

    Computes the log-determinant of the K matrix accumulated from activation patterns
    in the adapter modules (lightweight_mlp layers).

    Args:
        model: SAM model with adapter modules
        train_loader: DataLoader for training data
        arch_config: Architecture configuration dict

    Returns:
        tuple: (mean_loss, naswot_score, empty_dict)
    """
    global K_matrix, local_rank, activation_patterns

    # Get distributed training info
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    device = next(model.parameters()).device

    # Reset global state for fresh computation
    K_matrix = None
    activation_patterns = []

    model.train()
    loss_list = []

    # Progress bar only on rank 0
    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='naswot')
    else:
        pbar = None

    # Register hooks on Linear layers within adapter modules
    # Note: We target Linear layers instead of activation functions because
    # the activation function object is shared across all 12 layers (same nn.GELU/nn.ReLU instance)
    # By hooking Linear layers, we get 12 separate hooks - one per adapter layer
    hooks = []

    for name, module in model.named_modules():
        # Target Linear layers within lightweight_mlp (e.g., "lightweight_mlp_0.0", "lightweight_mlp_1.0", ...)
        is_adapter_linear = "prompt_generator" in name and "lightweight_mlp_" in name and isinstance(module, nn.Linear)

        if is_adapter_linear:
            hooks.append(module.register_forward_hook(counting_forward_hook))

    # Iterate through all batches (like ZiCo) to accumulate K matrix
    for _, batch in enumerate(train_loader):
        for k, v in batch.items():
            batch[k] = v.to(device)

        model.zero_grad()
        inp = batch['inp']
        gt = batch['gt']

        # Use SAM model's set_input/search_backward pattern
        model.set_input(inp, gt)
        model.search_backward()

        # Collect loss for consistency with ZiCo interface
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)

        if pbar is not None:
            pbar.update(1)

    # Remove hooks to prevent memory leak
    for hook in hooks:
        hook.remove()

    if pbar is not None:
        pbar.close()

    # Calculate NASWOT score
    score = caculate_naswot(max_samples=64)

    # Return format consistent with getzico: (mean_loss, score, score_dict)
    loss = [l.item() for l in loss_list]
    mean_loss = sum(loss) / len(loss) if loss else 0.0

    return mean_loss, score, {}