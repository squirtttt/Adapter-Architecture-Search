import argparse
import ast
import json
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


SEARCH_LINE_RE = re.compile(r"^\[op-cond-search\s+(\d+)/(\d+)\]\s+(.*)$")
PART_RE = re.compile(
    r"op=(?P<op>[^,]+),\s+config=(?P<config>\{.*?\})\s+"
    r"gamma=(?P<gamma>\[.*?\])\s+"
    r"(?P<proxy_name>ZiCo|zico|naswot|NASWOT)=(?P<proxy>[-+0-9.eE]+)\s+"
    r"score=(?P<score>[-+0-9.eE]+)\s+"
    r"params=(?P<params>\d+)"
    r"(?:\s+gmacs=(?P<gmacs>[-+0-9.eE]+))?"
)


def parse_search_log(log_path: str) -> List[Dict]:
    """Parse v2 operation-conditioned search logs into candidate records."""
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

                config = ast.literal_eval(match.group("config"))
                gamma = ast.literal_eval(match.group("gamma"))
                records.append(
                    {
                        "iteration": iteration,
                        "sample": sample_idx,
                        "operation": match.group("op"),
                        "config": config,
                        "gamma": [float(v) for v in gamma],
                        "proxy_name": match.group("proxy_name"),
                        "proxy_score": float(match.group("proxy")),
                        "penalized_score": float(match.group("score")),
                        "params": int(match.group("params")),
                        "gmacs": float(match.group("gmacs")) if match.group("gmacs") is not None else None,
                    }
                )
    if not records:
        raise RuntimeError(f"No op-cond-search records were parsed from {log_path}")
    return records


def load_final_result(search_dir: str) -> Dict:
    json_path = os.path.join(search_dir, "searched_operation_conditioned_zaas_config.json")
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        return json.load(f)


def plot_best_score_trajectory(records: List[Dict], out_path: str):
    iterations = sorted({r["iteration"] for r in records})
    iter_best = []
    running_best = []
    best_so_far = -np.inf
    for it in iterations:
        best = max(r["penalized_score"] for r in records if r["iteration"] == it)
        best_so_far = max(best_so_far, best)
        iter_best.append(best)
        running_best.append(best_so_far)

    plt.figure(figsize=(7, 4))
    plt.plot(iterations, iter_best, marker="o", label="Iteration best")
    plt.plot(iterations, running_best, marker="s", label="Running best")
    plt.xlabel("Search iteration")
    plt.ylabel("Penalized score")
    plt.title("Best Score Trajectory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_operation_timeline(records: List[Dict], out_path: str):
    """Plot sampled operations while making score quality visually explicit."""
    operations = sorted({r["operation"] for r in records})
    op_to_y = {op: i for i, op in enumerate(operations)}
    scores = np.asarray([r["penalized_score"] for r in records], dtype=float)
    score_min = float(scores.min())
    score_max = float(scores.max())
    score_span = max(score_max - score_min, 1e-8)
    best_record = max(records, key=lambda r: r["penalized_score"])

    plt.figure(figsize=(9, 4.8))
    xs = [r["iteration"] for r in records]
    ys = [op_to_y[r["operation"]] for r in records]
    color_values = [r["penalized_score"] for r in records]
    sizes = [45 + 260 * ((r["penalized_score"] - score_min) / score_span) for r in records]

    scatter = plt.scatter(
        xs,
        ys,
        c=color_values,
        s=sizes,
        cmap="viridis",
        vmin=score_min,
        vmax=score_max,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.82,
        label="sampled candidate",
    )

    iteration_best = get_iteration_best_records(records)
    plt.plot(
        [r["iteration"] for r in iteration_best],
        [op_to_y[r["operation"]] for r in iteration_best],
        color="white",
        linewidth=3.5,
        alpha=0.85,
        zorder=2,
    )
    plt.plot(
        [r["iteration"] for r in iteration_best],
        [op_to_y[r["operation"]] for r in iteration_best],
        color="black",
        marker="o",
        markersize=5,
        linewidth=1.5,
        label="iteration best",
        zorder=3,
    )

    plt.scatter(
        best_record["iteration"],
        op_to_y[best_record["operation"]],
        s=360,
        marker="*",
        color="gold",
        edgecolor="black",
        linewidth=0.9,
        label="global best",
        zorder=5,
    )
    config_text = ", ".join(f"{k}={v}" for k, v in best_record["config"].items())
    plt.text(
        0.98,
        0.97,
        f"Global best\noperation: {best_record['operation']}\nconfig: {config_text}\nscore: {best_record['penalized_score']:.3f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.4", "alpha": 0.92},
    )

    plt.yticks(range(len(operations)), operations)
    plt.xlabel("Search iteration")
    plt.ylabel("Sampled operation")
    plt.title("Operation Selection Timeline with Penalized Score")
    cbar = plt.colorbar(scatter, fraction=0.035, pad=0.025)
    cbar.set_label("penalized score")
    plt.grid(True, axis="x", alpha=0.25)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def get_iteration_best_records(records: List[Dict]) -> List[Dict]:
    iteration_best = []
    for iteration in sorted({r["iteration"] for r in records}):
        candidates = [r for r in records if r["iteration"] == iteration]
        iteration_best.append(max(candidates, key=lambda r: r["penalized_score"]))
    return iteration_best


def plot_gamma_heatmap(records: List[Dict], out_path: str):
    """Show how the best gamma pattern changes at each search iteration."""
    iteration_best = get_iteration_best_records(records)
    gamma = np.asarray([r["gamma"] for r in iteration_best], dtype=float)
    y_labels = [str(r["iteration"]) for r in iteration_best]

    height = max(3.2, 0.36 * len(iteration_best) + 1.4)
    plt.figure(figsize=(8, height))
    im = plt.imshow(gamma, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.xticks(range(gamma.shape[1]), [str(i) for i in range(gamma.shape[1])])
    plt.yticks(range(len(y_labels)), y_labels)
    plt.xlabel("Transformer layer")
    plt.ylabel("Search iteration")
    plt.title("Iteration-wise Best Candidate Gamma")
    cbar = plt.colorbar(im, fraction=0.04, pad=0.03)
    cbar.set_label("gamma")
    if len(iteration_best) <= 20:
        for row_idx in range(gamma.shape[0]):
            for layer_idx, value in enumerate(gamma[row_idx]):
                text_color = "white" if value < 0.55 else "black"
                plt.text(layer_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_final_gamma_heatmap(records: List[Dict], final_result: Dict, out_path: str):
    best_record = max(records, key=lambda r: r["penalized_score"])
    gamma = final_result.get("gamma") or final_result.get("best_sampled", {}).get("gamma") or best_record["gamma"]
    gamma = np.asarray(gamma, dtype=float)[None, :]

    plt.figure(figsize=(8, 2.4))
    im = plt.imshow(gamma, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.xticks(range(gamma.shape[1]), [str(i) for i in range(gamma.shape[1])])
    plt.yticks([0], ["best"])
    plt.xlabel("Transformer layer")
    plt.title("Final Best Gamma")
    cbar = plt.colorbar(im, fraction=0.04, pad=0.03)
    cbar.set_label("gamma")
    for layer_idx, value in enumerate(gamma[0]):
        text_color = "white" if value < 0.55 else "black"
        plt.text(layer_idx, 0, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_score_vs_params(records: List[Dict], out_path: str):
    operations = sorted({r["operation"] for r in records})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(operations), 1)))
    op_to_color = {op: colors[i % len(colors)] for i, op in enumerate(operations)}

    plt.figure(figsize=(7, 4.8))
    for op in operations:
        op_records = [r for r in records if r["operation"] == op]
        x = np.asarray([r["params"] for r in op_records], dtype=float) / 1e6
        y = np.asarray([r["penalized_score"] for r in op_records], dtype=float)
        gamma_sum = np.asarray([sum(r["gamma"]) for r in op_records], dtype=float)
        sizes = 35 + 18 * gamma_sum
        plt.scatter(
            x,
            y,
            s=sizes,
            color=op_to_color[op],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.75,
            label=op,
        )

    best_record = max(records, key=lambda r: r["penalized_score"])
    plt.scatter(
        best_record["params"] / 1e6,
        best_record["penalized_score"],
        s=230,
        marker="*",
        color="gold",
        edgecolor="black",
        linewidth=0.8,
        label="best sampled",
        zorder=10,
    )
    plt.xlabel("Adapter parameters (M)")
    plt.ylabel("Penalized score")
    plt.title("Score vs Adapter Parameters")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize v2 operation-conditioned ZAAS search logs.")
    parser.add_argument("--search_dir", required=True, help="Directory containing log.txt and search result JSON.")
    parser.add_argument("--out_dir", default=None, help="Directory to save figures. Defaults to <search_dir>/figures.")
    args = parser.parse_args()

    search_dir = args.search_dir
    out_dir = args.out_dir or os.path.join(search_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    records = parse_search_log(os.path.join(search_dir, "log.txt"))
    final_result = load_final_result(search_dir)

    plot_best_score_trajectory(records, os.path.join(out_dir, "best_score_trajectory.png"))
    plot_operation_timeline(records, os.path.join(out_dir, "operation_selection_timeline.png"))
    plot_gamma_heatmap(records, os.path.join(out_dir, "gamma_heatmap.png"))
    plot_final_gamma_heatmap(records, final_result, os.path.join(out_dir, "final_gamma_heatmap.png"))
    plot_score_vs_params(records, os.path.join(out_dir, "score_vs_params.png"))

    summary_path = os.path.join(out_dir, "visualization_summary.json")
    best = max(records, key=lambda r: r["penalized_score"])
    with open(summary_path, "w") as f:
        json.dump(
            {
                "num_candidates": len(records),
                "num_iterations": len({r["iteration"] for r in records}),
                "best_sampled_from_log": best,
                "final_result": final_result,
            },
            f,
            indent=2,
        )

    print(f"Saved figures to: {out_dir}")
    print(f"Best sampled: {best['operation']} {best['config']} score={best['penalized_score']:.4f}")


if __name__ == "__main__":
    main()
