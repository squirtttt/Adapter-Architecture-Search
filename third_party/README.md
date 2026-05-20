# Official External Repositories

This directory is used for official upstream code that is intentionally kept
outside the local AAS implementation.

Run:

```bash
bash scripts/setup_official_sam2_repos.sh
```

Expected clones:

- `third_party/sam2`: official Meta SAM2, <https://github.com/facebookresearch/sam2>
- `third_party/SAM2-Adapter-PyTorch`: official SAM2-Adapter branch,
  <https://github.com/tianrun-chen/SAM-Adapter-PyTorch/tree/SAM2-Adapter-for-Segment-Anything-2>

The cloned repos are ignored by git so local upstream checkouts and generated
files do not pollute the AAS repository history.
