# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **SAM-Adapter Architecture Search** project that performs Neural Architecture Search (NAS) on adapter modules for the Segment Anything Model (SAM). The goal is to find optimal adapter configurations for domain-specific segmentation tasks (camouflaged object detection, polyp segmentation, etc.).

## Common Commands

### Architecture Search (Distributed Training)
```bash
# Basic search with default config
torchrun --nproc_per_node=<NUM_GPUS> search_up.py --config ./configs/search_demo.yaml --name <SAVE_NAME>

# Search with specific proxy method
torchrun --nproc_per_node=<NUM_GPUS> search_up.py --config ./configs/search_polyp.yaml --proxy zico --name search_polyp

# Legacy search script (random sampling approach)
torchrun --nproc_per_node=<NUM_GPUS> search.py --config ./configs/search_demo.yaml --name <SAVE_NAME>
```

### Training
```bash
torchrun --nproc_per_node=<NUM_GPUS> train.py --config <CONFIG_PATH> --name <SAVE_NAME>
```

### Testing
```bash
python test.py --config <CONFIG_PATH> --model <MODEL_PATH>
```

## Architecture

### Core Components

1. **SAM Model** (`models/sam.py`): Wrapper around SAM's ImageEncoderViT and MaskDecoder
   - `set_input()`: Set input image and ground truth mask
   - `optimize_parameters()`: Training step with gradient update
   - `search_backward()`: Gradient computation without weight update (for NAS scoring)
   - `infer()`: Inference mode

2. **Prompt Generator** (`models/mmseg/models/sam/image_encoder.py`):
   - Lightweight MLP adapters inserted into ViT blocks
   - Search space includes: `scale_factor`, `prompt_activation`, `alpha` (layer-wise inclusion weights)

3. **Registry Pattern**: Both `models/` and `datasets/` use a decorator-based registration system
   - `@register('name')` decorator registers classes
   - `models.make(spec)` / `datasets.make(spec)` instantiates from config specs

### Search Scripts

- **`search_up.py`**: Primary search script with iterative alpha optimization
  - Uses perturbation-based search with softmax-weighted direction updates
  - Supports early stopping with patience
  - Alpha values control per-layer adapter inclusion (sigmoid → threshold → binary mask)

- **`search.py`**: Legacy random sampling search
  - Generates all candidate architectures upfront
  - Samples 1000 random architectures for evaluation

### Zero-Shot Proxies (`ZeroShotProxy/`)
- **ZICO** (`compute_zico.py`): Default proxy, computes gradient-based score
- **NASWOT** (`compute_naswot.py`): Alternative proxy
- **ZEN** (`compute_zen.py`): Alternative proxy (placeholder)

### Config Structure (YAML)
```yaml
train_dataset / val_dataset / test_dataset:
  dataset: {name, args}
  wrapper: {name, args}
  batch_size: int

model:
  name: sam
  args:
    encoder_mode:
      scale_factor: int      # MLP bottleneck reduction
      prompt_activation: str # GELU or ReLU
      alpha: list[12]        # Per-layer adapter weights
      embed_dim: int         # ViT embedding dimension
      depth: int             # Number of transformer blocks

search:
  patient: int       # Early stopping patience
  iteration: int     # Steps per architecture candidate
  sample_size: int   # Number of perturbations (K)
  epsilon: float     # Perturbation magnitude
  tau: float         # Softmax temperature
  delta: float       # Minimum score improvement
  threshold: float   # Alpha binarization threshold
```

### Key Data Flow
1. Config YAML specifies dataset, model architecture, and search hyperparameters
2. Search script generates adapter candidates (scale_factor × activation combinations)
3. For each candidate, alpha values are optimized via perturbation search
4. ZICO score computed using gradients from `prompt_generator` layers
5. Best architecture saved to `save/<name>/best_arch.yaml`

## Important Notes

- SAM requires **1024×1024 input images** (256×256 not supported)
- All training uses **DistributedDataParallel** with NCCL backend
- Pretrained SAM checkpoint expected at `./pretrained/sam_vit_b_01ec64.pth`
- Results saved to `./save/<experiment_name>/`
- Only `prompt_generator` parameters are trainable; rest of image_encoder is frozen
