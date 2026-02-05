#!/usr/bin/env python3
"""
Generate LaTeX tables for the paper from MTEB result JSONs in ./results.

Outputs:
  tables/mteb_category_averages.tex
  tables/mteb_detailed.tex
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path


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


def get_model_results(base_dir: str = "results") -> dict[str, dict]:
    model_data: dict[str, dict] = {}
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory '{base_dir}' not found.")

    for model_dir_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_dir_name)
        if not os.path.isdir(model_path):
            continue

        clean_name = model_dir_name.split("__")[-1] if "__" in model_dir_name else model_dir_name

        revisions = [
            d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))
        ]
        if not revisions:
            continue

        revisions.sort(key=lambda x: os.path.getmtime(os.path.join(model_path, x)), reverse=True)
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


def write_category_table(out_path: Path, data: dict[str, dict], model_cols: list[tuple[str, str]]) -> None:
    categories = ["BitextMining", "Classification", "Clustering", "Other", "Pair Classification", "Retrieval", "STS"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Category} & " + " & ".join(f"\\textbf{{{label}}}" for _, label in model_cols) + " \\\\")
    lines.append("\\midrule")

    for cat in categories:
        row = [cat]
        avgs: list[float] = []
        for model_key, _label in model_cols:
            tasks = data[model_key]["tasks"]
            scores = [t["score"] for t in tasks if t["category"] == cat]
            avgs.append((sum(scores) / len(scores)) if scores else float("nan"))

        best = max([a for a in avgs if a == a], default=float("nan"))
        for avg in avgs:
            if avg != avg:
                row.append("-")
            elif best == best and abs(avg - best) < 1e-9:
                row.append(f"\\textbf{{{avg:.2f}}}")
            else:
                row.append(f"{avg:.2f}")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_detailed_table(out_path: Path, data: dict[str, dict], model_cols: list[tuple[str, str]]) -> None:
    # Group tasks across all models by category + task name.
    categories = ["BitextMining", "Classification", "Clustering", "Other", "Pair Classification", "Retrieval", "STS"]

    tasks_by_model: dict[str, dict[str, float]] = {}
    cat_by_task: dict[str, str] = {}
    for model_key, _ in model_cols:
        m = {}
        for t in data[model_key]["tasks"]:
            m[t["task_name"]] = float(t["score"])
            cat_by_task[t["task_name"]] = t["category"]
        tasks_by_model[model_key] = m

    tasks_in_cat: dict[str, list[str]] = defaultdict(list)
    for task_name, cat in cat_by_task.items():
        tasks_in_cat[cat].append(task_name)
    for cat in tasks_in_cat:
        tasks_in_cat[cat].sort()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Task} & " + " & ".join(f"\\textbf{{{label}}}" for _, label in model_cols) + " \\\\")
    lines.append("\\midrule")

    for cat in categories:
        if cat not in tasks_in_cat:
            continue
        lines.append(f"\\multicolumn{{{1+len(model_cols)}}}{{l}}{{\\textit{{{cat}}}}} \\\\")
        for task_name in tasks_in_cat[cat]:
            row = [task_name]
            scores = []
            for model_key, _ in model_cols:
                val = tasks_by_model[model_key].get(task_name)
                scores.append(val if val is not None else float("nan"))
            best = max([s for s in scores if s == s], default=float("nan"))
            for s in scores:
                if s != s:
                    row.append("-")
                elif best == best and abs(s - best) < 1e-9:
                    row.append(f"\\textbf{{{s:.2f}}}")
                else:
                    row.append(f"{s:.2f}")
            lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = get_model_results()

    # Model columns in paper order
    model_cols = [
        ("mft-random", "TurkishTokenizer"),
        ("newmindaiMursit-random", "Mursit"),
        ("cosmosGPT2-random", "CosmosGPT2"),
        ("tabi-random", "Tabi"),
    ]
    missing = [k for k, _ in model_cols if k not in data]
    if missing:
        raise RuntimeError(f"Missing model results for: {missing}. Found: {sorted(data.keys())}")

    write_category_table(Path("tables/mteb_category_averages.tex"), data, model_cols)
    print("✓ Wrote tables/mteb_category_averages.tex")

    write_detailed_table(Path("tables/mteb_detailed.tex"), data, model_cols)
    print("✓ Wrote tables/mteb_detailed.tex")


if __name__ == "__main__":
    main()
