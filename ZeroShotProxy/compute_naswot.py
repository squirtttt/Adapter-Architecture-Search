import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.distributed as dist


# Global variables for K matrix accumulation
K_matrix = None
local_rank = None


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



def counting_forward_hook(module, inp, out):
    # K 행렬을 저장하기 위해 model.K를 사용한다고 가정
    global K_matrix # global 변수 또는 model 객체 속성 사용

    try:
        if not module.visited_backwards:
            return
        if isinstance(inp, tuple):
            inp = inp[0]
        # 입력 텐서 평탄화 및 이진화 (ReLU/GELU의 입력값 I > 0)
        inp = inp.view(inp.size(0), -1)
        x = (inp > 0).float() 
        
        # 상관 행렬 K 누적
        K_add = x @ x.t()
        K2_add = (1. - x) @ (1. - x.t())
        
        # K_matrix에 누적
        if K_matrix is None:
            K_matrix = (K_add + K2_add).cpu().numpy()
        else:
            K_matrix += (K_add + K2_add).cpu().numpy()
            
    except Exception as err:
        print('---- error on module: ', type(module))
        raise err


# 역전파 완료를 표시하는 Backward Hook
def counting_backward_hook(module, inp, out):
    module.visited_backwards = True

# Jacobian 계산을 통해 K 행렬 누적을 유도하는 함수
def get_batch_jacobian_for_naswot(net, x):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x) # 순전파: Forward Hook 호출 준비
    y.backward(torch.ones_like(y)) # 역전파: Backward Hook 호출, Forward Hook 실행
    # Jacobian (x.grad)는 NASWOT 스코어에 직접 사용되지 않음.
    # 하지만 역전파 자체가 K 행렬 누적을 트리거하는 필수 단계임.
    x.grad.detach() # 기울기 detach
    return y.detach()



def caculate_naswot():
    global K_matrix
    
    # K 행렬에 대해 LogDet 계산
    if K_matrix is None:
        return -1000.0 # K 행렬이 누적되지 않은 경우 낮은 값 반환
        
    score = logdet(K_matrix)
    
    # 다음 NASWOT 계산을 위해 K_matrix 초기화
    K_matrix = None 
    
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
    global K_matrix, local_rank

    # Get distributed training info
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    device = next(model.parameters()).device

    # Reset K_matrix for fresh computation
    K_matrix = None

    model.train()
    loss_list = []

    # Progress bar only on rank 0
    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='naswot')
    else:
        pbar = None

    # Register hooks on adapter activation functions
    ACTIVATION_TYPES = (nn.ReLU, nn.GELU)
    hooks = []

    for name, module in model.named_modules():
        # Only target activation functions within adapter modules (prompt_generator)
        is_adapter_module = "prompt_generator" in name and "lightweight_mlp_" in name
        is_activation = isinstance(module, ACTIVATION_TYPES)

        if is_adapter_module and is_activation:
            module.visited_backwards = False
            hooks.append(module.register_forward_hook(counting_forward_hook))
            hooks.append(module.register_backward_hook(counting_backward_hook))

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
    score = caculate_naswot()

    # Return format consistent with getzico: (mean_loss, score, score_dict)
    loss = [l.item() for l in loss_list]
    mean_loss = sum(loss) / len(loss) if loss else 0.0

    return mean_loss, score, {}