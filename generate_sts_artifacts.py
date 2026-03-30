#!/usr/bin/env python3
"""
Generate STS artifacts (figure + LaTeX table) from sts_benchmark_results.json.

This script intentionally does NOT re-run the benchmark (which requires network access
to fetch figenfikri/stsb_tr). It only visualizes the already-recorded results.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt


MODEL_LABELS = {
    "alibayram/mft-random": "TurkishTokenizer",
    "alibayram/newmindaiMursit-random": "Mursit",
    "alibayram/cosmosGPT2-random": "CosmosGPT2",
    "alibayram/tabi-random": "Tabi",
}

MODEL_ORDER = [
    "alibayram/mft-random",
    "alibayram/newmindaiMursit-random",
    "alibayram/cosmosGPT2-random",
    "alibayram/tabi-random",
]


def parse_iso(ts: str) -> datetime:
    # Python can parse the `datetime.now().isoformat()` we write in the repo.
    return datetime.fromisoformat(ts)


def load_latest_by_split(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raw = [raw]

    latest: dict[str, dict] = {}
    for entry in raw:
        ts = parse_iso(entry["timestamp"])
        for res in entry.get("results", []):
            split = res["split"]
            cur = latest.get(split)
            if cur is None or ts > cur["_timestamp"]:
                latest[split] = {"_timestamp": ts, **res}
            else:
                # latest already has a newer timestamp; but we still need per-model values.
                pass

    # The JSON structure stores multiple models per timestamp; reconstruct per split.
    per_split: dict[str, dict] = {}
    for entry in raw:
        ts = parse_iso(entry["timestamp"])
        for split in set(r["split"] for r in entry.get("results", [])):
            # only accept the latest timestamp for that split
            if split not in latest or ts != latest[split]["_timestamp"]:
                continue
            per_split[split] = {
                "timestamp": ts,
                "dataset": entry.get("dataset"),
                "results": entry.get("results", []),
            }
    return per_split


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    # 95% CI with Fisher z-transform.
    if n <= 3:
        return (float("nan"), float("nan"))
    z = math.atanh(max(min(r, 0.999999), -0.999999))
    se = 1.0 / math.sqrt(n - 3)
    zcrit = 1.959963984540054  # ~= scipy.stats.norm.ppf(0.975)
    lo = math.tanh(z - zcrit * se)
    hi = math.tanh(z + zcrit * se)
    return (lo, hi)


def write_latex_table(out_path: Path, split_rows: dict[str, list[dict]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v: float, best: float) -> str:
        return f"\\textbf{{{v:.2f}}}" if abs(v - best) < 1e-9 else f"{v:.2f}"

    def row(model_id: str, split: str, pearson: float, spearman: float, best_p: float, best_s: float) -> str:
        label = MODEL_LABELS.get(model_id, model_id)
        return f"{label} & {split} & {fmt(pearson, best_p)} & {fmt(spearman, best_s)} \\\\"

    lines: list[str] = []
    lines.append("\\begin{tabular}{llrr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Split} & \\textbf{Pearson} & \\textbf{Spearman} \\\\")
    lines.append("\\midrule")

    for split in ["test", "train"]:
        results = split_rows.get(split, [])
        by_model = {r["model"]: r for r in results}
        present = [by_model[m] for m in MODEL_ORDER if m in by_model]
        best_p = max((float(r["pearson"]) for r in present), default=float("nan")) * 100.0
        best_s = max((float(r["spearman"]) for r in present), default=float("nan")) * 100.0
        for model_id in MODEL_ORDER:
            if model_id not in by_model:
                continue
            r = by_model[model_id]
            lines.append(
                row(
                    model_id=model_id,
                    split=split,
                    pearson=float(r["pearson"]) * 100.0,
                    spearman=float(r["spearman"]) * 100.0,
                    best_p=best_p,
                    best_s=best_s,
                )
            )

        if split == "test" and "train" in split_rows:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_split_bar(results: list[dict], title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_model = {r["model"]: r for r in results}
    models = [m for m in MODEL_ORDER if m in by_model]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    pearson = [float(by_model[m]["pearson"]) * 100.0 for m in models]
    spearman = [float(by_model[m]["spearman"]) * 100.0 for m in models]

    # Styling consistent with the MTEB plots in this repo.
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    palette = {
        "TurkishTokenizer": "#3498db",
        "Mursit": "#9b59b6",
        "CosmosGPT2": "#2ecc71",
        "Tabi": "#e74c3c",
    }

    fig, ax = plt.subplots(figsize=(10.0, 3.9))
    y = list(range(len(labels)))
    y_pearson = [i - 0.12 for i in y]
    y_spearman = [i + 0.12 for i in y]

    # Subtle per-model color on the connecting line; points colored by metric.
    for i, label in enumerate(labels):
        line_color = palette.get(label, "#7f8c8d")
        ax.plot([pearson[i], spearman[i]], [y_pearson[i], y_spearman[i]], color=line_color, alpha=0.25, linewidth=2)

    ax.scatter(pearson, y_pearson, color="#1f77b4", marker="o", s=55, label="Pearson")
    ax.scatter(spearman, y_spearman, color="#ff7f0e", marker="D", s=48, label="Spearman")

    # Value labels
    for i in range(len(labels)):
        ax.text(pearson[i] + 0.4, y_pearson[i], f"{pearson[i]:.2f}", va="center", fontsize=10)
        ax.text(spearman[i] + 0.4, y_spearman[i], f"{spearman[i]:.2f}", va="center", fontsize=10)

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Correlation (%)")
    ax.set_title(title, fontweight="bold", pad=10)

    max_val = max(max(pearson), max(spearman))
    ax.set_xlim(0, max(60, max_val + 6))
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.grid(False, axis="y")
    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    src = Path("sts_benchmark_results.json")
    if not src.exists():
        raise FileNotFoundError("sts_benchmark_results.json not found.")

    per_split = load_latest_by_split(src)
    if not per_split:
        raise RuntimeError("No STS results found in sts_benchmark_results.json.")

    # LaTeX table used by the paper.
    split_rows = {k: v["results"] for k, v in per_split.items()}
    write_latex_table(Path("tables/sts_results.tex"), split_rows)
    print("✓ Wrote tables/sts_results.tex")

    # Figures
    if "test" in per_split:
        plot_split_bar(
            per_split["test"]["results"],
            "STS benchmark performance (test split)",
            Path("paper/figures/sts_benchmark_chart_test.png"),
        )
        print("✓ Wrote paper/figures/sts_benchmark_chart_test.png")

    if "train" in per_split:
        plot_split_bar(
            per_split["train"]["results"],
            "STS benchmark performance (train split)",
            Path("paper/figures/sts_benchmark_chart_train.png"),
        )
        print("✓ Wrote paper/figures/sts_benchmark_chart_train.png")

    # Quick CI printouts for paper text convenience.
    if "test" in per_split:
        by_model = {r["model"]: r for r in per_split["test"]["results"]}
        if "alibayram/mft-random" in by_model:
            r = float(by_model["alibayram/mft-random"]["pearson"])
            n = int(by_model["alibayram/mft-random"]["num_samples"])
            lo, hi = fisher_ci(r, n)
            print(f"TurkishTokenizer test Pearson 95% CI: [{lo*100:.2f}, {hi*100:.2f}] (n={n})")
        if "alibayram/tabi-random" in by_model:
            r = float(by_model["alibayram/tabi-random"]["pearson"])
            n = int(by_model["alibayram/tabi-random"]["num_samples"])
            lo, hi = fisher_ci(r, n)
            print(f"Tabi test Pearson 95% CI: [{lo*100:.2f}, {hi*100:.2f}] (n={n})")


if __name__ == "__main__":
    main()
