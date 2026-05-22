import argparse
import ast
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import yaml


SEARCH_LINE_RE = re.compile(r"^\[op-cond-search\s+(\d+)/(\d+)\]\s+(.*)$")
PART_RE = re.compile(
    r"op=(?P<op>[^,]+),\s+config=(?P<config>\{.*?\})\s+"
    r"gamma=(?P<gamma>\[.*?\])\s+"
    r"(?P<proxy_name>ZiCo|zico|naswot|NASWOT)=(?P<proxy>[-+0-9.eE]+)\s+"
    r"score=(?P<score>[-+0-9.eE]+)\s+"
    r"params=(?P<params>\d+)"
)


def parse_search_log(log_path: str) -> List[Dict]:
    records = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            line_match = SEARCH_LINE_RE.match(line)
            if not line_match:
                continue
            iteration = int(line_match.group(1))
            body = line_match.group(3)
            candidate_parts = [p.strip() for p in body.split("||") if p.strip().startswith("op=")]
            for sample_idx, part in enumerate(candidate_parts, start=1):
                match = PART_RE.search(part)
                if not match:
                    continue
                records.append(
                    {
                        "iteration": iteration,
                        "sample": sample_idx,
                        "operation": match.group("op"),
                        "config": ast.literal_eval(match.group("config")),
                        "gamma": [float(v) for v in ast.literal_eval(match.group("gamma"))],
                        "proxy_name": match.group("proxy_name"),
                        "proxy_score": float(match.group("proxy")),
                        "penalized_score": float(match.group("score")),
                        "params": int(match.group("params")),
                    }
                )
    if not records:
        raise RuntimeError(f"No search candidate records found in {log_path}")
    return records


def read_search_hparams(search_dir: str) -> Dict:
    config_path = os.path.join(search_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("search", {})


def format_config(config: Dict) -> str:
    if not config:
        return "{}"
    return ", ".join(f"{k}={v}" for k, v in config.items())


def collect_results(search_dirs: List[str]) -> List[Dict]:
    results = []
    for search_dir in search_dirs:
        records = parse_search_log(os.path.join(search_dir, "log.txt"))
        best = max(records, key=lambda r: r["penalized_score"])
        hparams = read_search_hparams(search_dir)
        results.append(
            {
                "search_dir": search_dir,
                "name": os.path.basename(search_dir.rstrip(os.sep)),
                "K": int(hparams.get("K", -1)),
                "N": int(hparams.get("N", -1)),
                "operation": best["operation"],
                "config": best["config"],
                "gamma": best["gamma"],
                "proxy_name": best["proxy_name"],
                "proxy_score": best["proxy_score"],
                "penalized_score": best["penalized_score"],
                "params": best["params"],
            }
        )
    return sorted(results, key=lambda x: x["K"])


def plot_architecture_gamma_comparison(results: List[Dict], out_path: str):
    gamma = np.asarray([r["gamma"] for r in results], dtype=float)
    row_labels = [f"K={r['K']}" for r in results]

    fig_width = 13.0
    fig_height = max(3.6, 0.85 * len(results) + 1.6)
    fig, (ax_heat, ax_text) = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [1.25, 1.1]},
    )

    im = ax_heat.imshow(gamma, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax_heat.set_xticks(range(gamma.shape[1]))
    ax_heat.set_xticklabels([str(i) for i in range(gamma.shape[1])])
    ax_heat.set_yticks(range(len(row_labels)))
    ax_heat.set_yticklabels(row_labels)
    ax_heat.set_xlabel("Transformer layer")
    ax_heat.set_ylabel("Perturbation number")
    ax_heat.set_title("Best Architecture Gamma by K")

    for row_idx in range(gamma.shape[0]):
        for layer_idx, value in enumerate(gamma[row_idx]):
            text_color = "white" if value < 0.55 else "black"
            ax_heat.text(layer_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.045, pad=0.03)
    cbar.set_label("gamma")

    ax_text.axis("off")
    ax_text.set_title("Best Sampled Architecture", loc="left")
    y = 0.98
    line_gap = 0.24 if len(results) <= 4 else 0.18
    for r in results:
        desc = (
            f"K={r['K']}  (N={r['N']})\n"
            f"op: {r['operation']}\n"
            f"config: {format_config(r['config'])}\n"
            f"{r['proxy_name']}: {r['proxy_score']:.4f}   score: {r['penalized_score']:.4f}\n"
            f"params: {r['params']:,}"
        )
        ax_text.text(
            0.0,
            y,
            desc,
            transform=ax_text.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.75", "alpha": 0.95},
        )
        y -= line_gap

    fig.suptitle("Architecture and Layer-wise Gamma Comparison", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare best architecture and gamma heatmaps across v2 search directories.")
    parser.add_argument("--dirs", nargs="+", required=True, help="Search result directories containing log.txt and config.yaml.")
    parser.add_argument("--out", default="search_gamma_arch_comparison.png", help="Output image path.")
    args = parser.parse_args()

    results = collect_results(args.dirs)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plot_architecture_gamma_comparison(results, args.out)

    print(f"Saved comparison figure to: {args.out}")
    for r in results:
        print(
            f"K={r['K']}: {r['operation']} {r['config']} "
            f"{r['proxy_name']}={r['proxy_score']:.4f} score={r['penalized_score']:.4f}"
        )


if __name__ == "__main__":
    main()
