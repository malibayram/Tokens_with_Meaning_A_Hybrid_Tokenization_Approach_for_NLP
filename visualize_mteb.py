#!/usr/bin/env python3
"""
Visualize MTEB Benchmark Results - Academic Style Charts
"""

import glob
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

# Set academic style
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


def categorize_task(task_name: str) -> str:
    tn = task_name.lower()
    if "retrieval" in tn or "corpus" in tn or "fact" in tn:
        return "Retrieval"
    if "clustering" in tn:
        return "Clustering"
    if "sts" in tn:
        return "STS"
    if "nli" in tn or "snli" in tn or "mnli" in tn:
        return "Pair Classification"
    if "classification" in tn or "sentiment" in tn or "irony" in tn:
        return "Classification"
    if "bitext" in tn:
        return "BitextMining"
    return "Other"


def extract_main_score(payload: dict) -> float | None:
    if "scores" not in payload:
        return None
    scores = payload["scores"]
    for split in ["test", "test_matched", "test_mismatched", "validation", "dev"]:
        if split in scores and scores[split]:
            first_res = scores[split][0]
            if "main_score" in first_res:
                return float(first_res["main_score"])
    return None


def load_mteb_results(base_dir: str = "results") -> dict:
    model_data: dict[str, dict] = {}

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory '{base_dir}' not found.")

    for model_dir_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_dir_name)
        if not os.path.isdir(model_path):
            continue

        clean_name = (
            model_dir_name.split("__")[-1] if "__" in model_dir_name else model_dir_name
        )

        revisions = [
            d
            for d in os.listdir(model_path)
            if os.path.isdir(os.path.join(model_path, d))
        ]
        if not revisions:
            continue

        revisions.sort(
            key=lambda x: os.path.getmtime(os.path.join(model_path, x)), reverse=True
        )
        latest_rev = revisions[0]
        rev_path = os.path.join(model_path, latest_rev)

        tasks = []
        for json_file in glob.glob(os.path.join(rev_path, "*.json")):
            filename = os.path.basename(json_file)
            if filename == "model_meta.json":
                continue
            with open(json_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            score = extract_main_score(payload)
            if score is None:
                continue
            task_name = payload.get("task_name", filename.replace(".json", ""))
            tasks.append(
                {
                    "task_name": task_name,
                    "category": categorize_task(task_name),
                    "score": score * 100.0,
                }
            )

        if tasks:
            model_data[clean_name] = {
                "revision": latest_rev,
                "full_name": model_dir_name,
                "tasks": tasks,
            }

    return model_data


def build_category_matrix(
    model_data: dict, model_order: list[str], categories: list[str]
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for model in model_order:
        tasks = model_data[model]["tasks"]
        values = []
        for cat in categories:
            cat_scores = [t["score"] for t in tasks if t["category"] == cat]
            values.append(float(np.mean(cat_scores)) if cat_scores else float("nan"))
        out[model] = values
    return out


def build_overall_scores(model_data: dict, model_order: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for model in model_order:
        scores = [t["score"] for t in model_data[model]["tasks"]]
        out[model] = float(np.mean(scores)) if scores else float("nan")
    return out


def main() -> None:
    model_data = load_mteb_results()

    # Prefer canonical model ordering in plots (fallback to whatever exists).
    available = set(model_data.keys())
    preferred = [
        "cosmosGPT2-random",
        "mft-random",
        "newmindaiMursit-random",
        "tabi-random",
    ]
    model_order = [m for m in preferred if m in available] + sorted(
        list(available - set(preferred))
    )

    # Map internal IDs to display names.
    display_name = {
        "cosmosGPT2-random": "cosmosGPT2",
        "mft-random": "TurkishTokenizer",
        "newmindaiMursit-random": "Mursit",
        "tabi-random": "TABI",
    }
    display_models = [display_name.get(m, m) for m in model_order]

    # Stable colors aligned with the canonical display names above.
    palette = {
        "cosmosGPT2": "#2ecc71",
        "TurkishTokenizer": "#3498db",
        "Mursit": "#9b59b6",
        "TABI": "#e74c3c",
    }
    colors = [palette.get(dm, "#7f8c8d") for dm in display_models]

    categories = [
        "BitextMining",
        "Classification",
        "Clustering",
        "Other",
        "Pair Classification",
        "Retrieval",
        "STS",
    ]
    cat_data_raw = build_category_matrix(model_data, model_order, categories)
    cat_data = {display_name.get(k, k): v for k, v in cat_data_raw.items()}
    overall_scores_raw = build_overall_scores(model_data, model_order)
    overall_scores = {display_name.get(k, k): v for k, v in overall_scores_raw.items()}

    os.makedirs("figures", exist_ok=True)

    # - Chart 1: Overall MTEB comparison -
    fig1, ax = plt.subplots(figsize=(7.2, 4.6))
    sorted_names = sorted(overall_scores.keys(), key=lambda n: overall_scores[n])
    sorted_scores = [overall_scores[n] for n in sorted_names]
    model_colors = [palette.get(n, "#7f8c8d") for n in sorted_names]

    bars = ax.barh(
        sorted_names,
        sorted_scores,
        color=model_colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.6,
    )
    for bar, score in zip(bars, sorted_scores):
        ax.text(
            score + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}%",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Average Score (%)")
    ax.set_title("Overall MTEB Performance", fontweight="bold", pad=12)
    ax.set_xlim(0, max(45, float(np.nanmax(sorted_scores)) + 6))

    os.makedirs(os.path.join("paper", "figures"), exist_ok=True)
    out_overall = os.path.join("paper", "figures", "mteb_comparison.jpg")
    fig1.tight_layout()
    fig1.savefig(out_overall, facecolor="white")
    plt.close(fig1)
    print(f"✓ Saved: {out_overall}")

    # - Chart 2: Category comparison -
    fig2, ax2 = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(categories))
    width = 0.2
    offsets = np.linspace(-1.5, 1.5, num=len(display_models))

    for i, (model, color) in enumerate(zip(display_models, colors)):
        values = cat_data.get(model, [float("nan")] * len(categories))
        ax2.bar(
            x + offsets[i] * width,
            values,
            width,
            label=model,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    ax2.set_xlabel("Task Category")
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Performance by Category", fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=25, ha="right")
    ax2.legend(loc="upper right", framealpha=0.95)
    ax2.set_ylim(0, 75)
    ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    out_cat = os.path.join("paper", "figures", "mteb_comparison_by_category.jpg")
    fig2.tight_layout()
    fig2.savefig(out_cat, facecolor="white")
    plt.close(fig2)
    print(f"✓ Saved: {out_cat}")


if __name__ == "__main__":
    main()
