# SAM2 Official Experiments

This project now supports three separate SAM2-related experiment tracks.

## 1. Official Meta SAM2 Baseline

Config:

```bash
configs/sam2_camo_baseline.yaml
```

Run with the local AAS training loop:

```bash
bash scripts/run_official_sam2_baseline_train.sh
```

This uses the official Meta SAM2 package from `third_party/sam2` when it is not
installed in the Python environment. Meta's current SAM2 repo requires a newer
environment than the original AAS setup: Python >= 3.10 and torch >= 2.5.1.

## 2. Official SAM2-Adapter Baseline

First fetch the upstream repositories:

```bash
bash scripts/setup_official_sam2_repos.sh
```

Then run the official SAM2-Adapter branch in its own working directory:

```bash
bash scripts/run_official_sam2_adapter_train.sh
```

This path uses the official `SAM2-Adapter-for-Segment-Anything-2` branch code.
Its Hiera backbone has `PromptGenerator` inside
`models/sam2/modeling/backbones/hieradet.py`, and training freezes
`image_encoder` parameters except `prompt_generator`.

The config expects the checkpoint:

```bash
pretrained/sam2_hiera_base_plus.pt
```

## 3. SAM2 + AAS Adapter Search

Config:

```bash
configs/search_sam2_demo_v2.yaml
```

Run:

```bash
bash scripts/run_aas_sam2_search.sh
```

This is not the official SAM2-Adapter baseline. It uses the official SAM2
image encoder as the backbone and applies this repository's adapter search
space to the SAM2 image encoder.
