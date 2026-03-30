#!/usr/bin/env python3
"""MTEB Benchmark Results Report Generator."""
import json
import glob
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt


def get_model_results(base_dir="results"):
    """
    Traverse results directory to find models and their scores.
    Structure: results/<model_name>/<revision>/*.json
    """
    model_data = {}

    if not os.path.isdir(base_dir):
        print(f"Directory '{base_dir}' not found.")
        return {}

    # Iterate over model directories
    for model_dir_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_dir_name)
        if not os.path.isdir(model_path):
            continue

        # Clean model name (remove prefix/suffix if needed, or keep as is)
        # e.g. alibayram__mft-downstream-task-embeddinggemma -> mft-downstream-task-embeddinggemma
        clean_name = model_dir_name
        if "__" in clean_name:
            clean_name = clean_name.split("__")[-1]

        # Find revisions (subdirectories)
        revisions = [
            d
            for d in os.listdir(model_path)
            if os.path.isdir(os.path.join(model_path, d))
        ]
        if not revisions:
            print(f"No revisions found for {model_dir_name}")
            continue

        # Pick the latest modified revision directory
        # (Assuming the most recent evaluation is the relevant one)
        revisions.sort(
            key=lambda x: os.path.getmtime(os.path.join(model_path, x)), reverse=True
        )
        latest_rev = revisions[0]
        rev_path = os.path.join(model_path, latest_rev)

        # Parse results in the revision directory
        tasks = []
        json_files = glob.glob(os.path.join(rev_path, "*.json"))

        for json_file in json_files:
            filename = os.path.basename(json_file)
            if filename == "model_meta.json":
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Check for main score
                # Usually in scores -> test -> [0] -> main_score
                # But sometimes structure varies widely in MTEB results files
                # We'll try a generic approach
                score = None

                # Try standard MTEB format
                if "scores" in data:
                    scores = data["scores"]
                    # Priority: test > validation > dev
                    for split in [
                        "test",
                        "test_matched",
                        "test_mismatched",
                        "validation",
                        "dev",
                    ]:
                        if split in scores and scores[split]:
                            first_res = scores[split][0]
                            if "main_score" in first_res:
                                score = first_res["main_score"]
                                break

                if score is not None:
                    tasks.append(
                        {
                            "task_name": data.get(
                                "task_name", filename.replace(".json", "")
                            ),
                            "score": score * 100,  # Convert to percentage
                            "filename": filename,
                        }
                    )
            except Exception as e:
                print(f"Error reading {json_file}: {e}")

        if tasks:
            # Calculate average
            avg_score = sum(t["score"] for t in tasks) / len(tasks)
            model_data[clean_name] = {
                "revision": latest_rev,
                "full_name": model_dir_name,
                "average_score": avg_score,
                "tasks": tasks,
            }

    return model_data


def format_table(headers, rows):
    """Format data as a Markdown table."""
    if not rows:
        return "No data available."

    col_widths = []
    for i, h in enumerate(headers):
        w = len(h)
        if rows:
            max_row_w = max(len(str(r[i])) for r in rows)
            w = max(w, max_row_w)
        col_widths.append(w)

    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"

    lines = [fmt.format(*headers), sep]
    for r in rows:
        lines.append(fmt.format(*[str(c) for c in r]))

    return "\n".join(lines)


def categorize_task(task_name):
    """Categorize MTEB tasks based on their names."""
    tn = task_name.lower()
    if "retrieval" in tn or "corpus" in tn or "fact" in tn:
        return "Retrieval"
    elif "clustering" in tn:
        return "Clustering"
    elif "sts" in tn:
        return "STS"  # Semantic Textual Similarity
    elif "nli" in tn or "snli" in tn or "mnli" in tn:
        return "Pair Classification"
    elif "classification" in tn or "sentiment" in tn or "irony" in tn:
        return "Classification"
    elif "bitext" in tn:
        return "BitextMining"
    else:
        return "Other"


def generate_charts(model_data):
    """Generate summary and per-task charts."""
    if not model_data:
        return

    output_files = []

    # 1. Average Score Comparison
    plt.figure(figsize=(12, 8))

    # Sort models by average score
    sorted_models = sorted(
        model_data.items(), key=lambda x: x[1]["average_score"], reverse=True
    )
    names = [m[0] for m in sorted_models]
    scores = [m[1]["average_score"] for m in sorted_models]

    # Colors: Highlight MTEB vs Tabi vs Random
    colors = []
    for name in names:
        if "random" in name.lower():
            colors.append("lightgray")
        elif "mft" in name.lower():
            colors.append("skyblue")
        elif "tabi" in name.lower():
            colors.append("lightgreen")
        else:
            colors.append("gray")

    bars = plt.bar(names, scores, color=colors, edgecolor="black", alpha=0.8)

    plt.title("Average MTEB Score Comparison", fontsize=16)
    plt.ylabel("Average Score (%)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.bar_label(bars, fmt="%.2f%%", padding=3)
    plt.tight_layout()

    os.makedirs(os.path.join("paper", "figures"), exist_ok=True)
    avg_chart = os.path.join("paper", "figures", "mteb_average_scores.png")
    plt.savefig(avg_chart)
    plt.close()
    output_files.append(avg_chart)
    print(f"Generated {avg_chart}")

    return output_files


def main():
    print("Scanning results directory...")
    data = get_model_results()

    if not data:
        print("No model results found.")
        return

    # Prepare Markdown Report
    lines = ["# MTEB Benchmark Results Report\n"]

    # 1. Gather all tasks and all models
    all_models = sorted(data.keys())
    all_tasks = set()
    for m in data.values():
        for t in m["tasks"]:
            all_tasks.add(t["task_name"])
    all_tasks = sorted(list(all_tasks))

    # - Table 1: All Tasks with Highlighting -
    lines.append("# 🏆 Detailed Task Results\n")

    # Header
    table1_header = ["Task", "Category"] + all_models
    table1_rows = []

    for task in all_tasks:
        row = [task, categorize_task(task)]
        scores = []
        for model in all_models:
            # Find score for this model & task
            m_tasks = data[model]["tasks"]
            matches = [t for t in m_tasks if t["task_name"] == task]
            if matches:
                scores.append(matches[0]["score"])
            else:
                scores.append(-1.0)  # Indicator for missing

        # Determine max score (ignoring missing)
        valid_scores = [s for s in scores if s >= 0]
        max_score = max(valid_scores) if valid_scores else -1

        for s in scores:
            if s < 0:
                row.append("-")
            else:
                s_str = f"{s:.2f}%"
                if s == max_score and max_score > 0:
                    row.append(f"**{s_str}**")
                else:
                    row.append(s_str)
        table1_rows.append(row)

    lines.append(format_table(table1_header, table1_rows))
    lines.append("\n")

    # - Table 2: Categorized Results -
    lines.append("# 📂 Categorized Results\n")

    # Calculate average score per category per model
    categories = sorted(list(set(categorize_task(t) for t in all_tasks)))
    table2_header = ["Category"] + all_models
    table2_rows = []

    for cat in categories:
        row = [cat]
        cat_avg_scores = []

        # First pass: calculate averages
        for model in all_models:
            m_tasks = data[model]["tasks"]
            cat_scores = [
                t["score"] for t in m_tasks if categorize_task(t["task_name"]) == cat
            ]

            if cat_scores:
                avg = sum(cat_scores) / len(cat_scores)
                cat_avg_scores.append(avg)
            else:
                cat_avg_scores.append(-1.0)

        # Find max
        valid_avgs = [s for s in cat_avg_scores if s >= 0]
        max_avg = max(valid_avgs) if valid_avgs else -1

        # Second pass: format
        for avg in cat_avg_scores:
            if avg < 0:
                row.append("-")
            else:
                s_str = f"{avg:.2f}%"
                if avg == max_avg and max_avg > 0:
                    row.append(f"**{s_str}**")
                else:
                    row.append(s_str)

        table2_rows.append(row)

    lines.append(format_table(table2_header, table2_rows))
    lines.append("\n")

    # - Table 3: Average of All (Summary) -
    lines.append("# 📊 Overall Average Scores\n")

    sorted_models_by_avg = sorted(
        data.items(), key=lambda x: x[1]["average_score"], reverse=True
    )
    table3_header = ["Model", "Average Score", "Tasks Evaluated"]
    table3_rows = []

    best_avg = sorted_models_by_avg[0][1]["average_score"]

    for name, info in sorted_models_by_avg:
        avg_str = f"{info['average_score']:.2f}%"
        if info["average_score"] == best_avg:
            name_fmt = f"**{name}**"
            avg_str = f"**{avg_str}**"
        else:
            name_fmt = name

        table3_rows.append([name_fmt, avg_str, len(info["tasks"])])

    lines.append(format_table(table3_header, table3_rows))
    lines.append("\n")

    # Charts (expected to exist; generated by visualize_mteb.py or copied in)
    lines.append("![Overall MTEB Comparison](paper/figures/mteb_comparison.jpg)\n")

    # Fallback average chart (generated here)
    generate_charts(data)
    lines.append("![Average MTEB Scores](paper/figures/mteb_average_scores.png)\n")

    # Write to file
    output_filename = "MTEB_BENCHMARK_RESULTS.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report generated: {output_filename}")


if __name__ == "__main__":
    main()
