# ZAMs: Zero-shot Adapter Architecture Search for Segmentation Foundation Models

ZAMs is a zero-shot adapter architecture search pipeline for segmentation foundation models such as SAM and SAM2. It searches lightweight residual adapter architectures without fully training every candidate, then fine-tunes only the selected adapter while keeping the foundation backbone frozen.

The current implementation supports operation-conditioned hierarchical perturbation search over:

- Adapter primitive type
- Operation-specific adapter configuration
- Layer-wise residual importance values
- Continuous residual or hard insertion modes
- ZiCo and NASWOT-style zero-shot proxy evaluation

The final searched model follows a single-adapter philosophy:

```text
x_l' = x_l + gamma_l * A_{op, config}(x_l)
```

One adapter candidate is selected and reused across the selected or weighted image-encoder layers.

## Supported Backbones

### SAM ViT-B

Use the `repeated_adapter_sam` model with the SAM ViT-B image encoder.

Example configs:

```text
configs/search_demo_v2.yaml
configs/search_polyp_v2.yaml
```

### SAM2

Use the official `facebookresearch/sam2` repository under `third_party/sam2`.

Example configs:

```text
configs/search_sam2_demo_v2.yaml
configs/search_sam2_polyp_v2.yaml
configs/search_sam2_smoke_v2.yaml
configs/search_sam2_polyp_smoke_v2.yaml
```

SAM2 adapter insertion supports:

```yaml
adapter_insertion: auto    # Try block-level insertion, fallback when needed
adapter_insertion: block   # Force SAM2 image-encoder block adapters
adapter_insertion: feature # Apply repeated adapters after SAM2 image features
```

## Adapter Search Space

ZAMs searches among lightweight residual adapter primitives:

```text
identity
mlp
gated_mlp
dwconv
low_rank
frequency
channel_attention
edge_aware
```

Each primitive has operation-conditioned configuration fields. For example:

```yaml
operation_search_space:
  mlp:
    dim: [16, 32, 64, 128]
    activation: [gelu, relu, silu]
  gated_mlp:
    dim: [16, 32, 64, 128]
    gate: [geglu, swiglu]
  low_rank:
    rank: [4, 8, 16]
  frequency:
    dim: [16, 32, 64]
    activation: [gelu, silu]
    freq_mode: [avg_highpass, laplacian, token_smoothing]
```

Invalid fields are not perturbed or updated. For example, when `low_rank` is selected, only `rank` logits are used.

## Search Algorithm

The default search strategy is operation-conditioned hierarchical perturbation:

1. Perturb operation logits `alpha_op`.
2. Decode one operation by argmax.
3. Perturb only the selected operation's valid configuration logits.
4. Perturb layer-wise residual logits `alpha_layer`.
5. Build a candidate model with the same adapter repeated across layers.
6. Evaluate the candidate using a zero-shot proxy.
7. Apply sparsity and parameter penalties.
8. Update search logits with score-weighted perturbations.
9. Export the best sampled architecture and the final alpha-decoded architecture.

The default final selection is the best sampled architecture.

## Environment Setup

### SAM ViT-B Environment

For the original SAM ViT-B experiments, use the existing AAS environment.

```bash
conda env update -n aas -f environment_aas.yml
conda activate aas
```

### SAM2 Environment

Official SAM2 requires a newer Python/PyTorch stack. Create a separate environment:

```bash
bash scripts/setup_sam2_env.sh aas-sam
conda activate aas-sam
```

Download official SAM2 checkpoints and link them under `pretrained/`:

```bash
bash scripts/prepare_sam2_checkpoints.sh
```

The default SAM2 configs expect:

```text
pretrained/sam2.1_hiera_base_plus.pt
```

## Running Search

### CAMO with SAM ViT-B

```bash
torchrun --nproc_per_node=1  search_up_v2.py \
  --config configs/search_demo_v2.yaml \
  --name v2_camo_search
```

### Kvasir-SEG with SAM ViT-B

```bash
torchrun --nproc_per_node=1  search_up_v2.py \
  --config configs/search_polyp_v2.yaml \
  --name v2_polyp_search
```

### CAMO with SAM2

```bash
torchrun --nproc_per_node=1  search_up_v2.py \
  --config configs/search_sam2_demo_v2.yaml \
  --name sam2_camo_search
```

### Kvasir-SEG with SAM2

```bash
torchrun --nproc_per_node=1  search_up_v2.py \
  --config configs/search_sam2_polyp_v2.yaml \
  --name sam2_polyp_search
```

## Training the Selected Adapter

After search, train the best sampled architecture:

```bash
torchrun --nproc_per_node=1 --master_port=29511 train_v2.py \
  --config save/v2_camo_search/best_arch_hierarchical_v2.yaml \
  --name v2_camo_search_train
```

For SAM2 polyp:

```bash
torchrun --nproc_per_node=1 --master_port=29611 train_v2.py \
  --config save/sam2_polyp_search/best_arch_hierarchical_v2.yaml \
  --name sam2_polyp_search_train
```

Only adapter parameters are trainable. The SAM or SAM2 image encoder backbone remains frozen.

## Evaluation

```bash
python test_v2.py \
  --config save/v2_camo_search_train/config.yaml \
  --model save/v2_camo_search_train/model_epoch_best.pth \
  --dataset_key val_dataset \
  --device cuda:0 \
  --verbose
```

For polyp datasets:

```bash
python test_v2.py \
  --config save/sam2_polyp_search_train/config.yaml \
  --model save/sam2_polyp_search_train/model_epoch_best.pth \
  --dataset_key test_dataset \
  --device cuda:0 \
  --verbose
```
