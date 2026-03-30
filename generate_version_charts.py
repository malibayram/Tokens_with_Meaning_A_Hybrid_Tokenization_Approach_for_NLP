import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import glob
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12


def load_data():
    data = []

    # Map filenames to readable model names
    model_map = {
        "version_eval_alibayram_mft_random.json": "mft-random",
        "version_eval_alibayram_tabi_random.json": "tabi-random",
        "version_eval_alibayram_cosmosgpt2_random.json": "cosmosGPT2-random",
        "version_eval_alibayram_newmindaimursit_random.json": "newmindaiMursit-random",
    }

    for filename, model_name in model_map.items():
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found.")
            continue

        with open(filename, "r") as f:
            content = json.load(f)

        # Extract results
        results = content.get("results", [])

        # Sort by commit date
        results.sort(key=lambda x: x["commit_date"])

        for i, res in enumerate(results):
            if "pearson" not in res or "spearman" not in res:
                # print(f"Skipping index {i} for {model_name} due to missing metrics.")
                continue

            # if commit date is earlier 2026-02-04T00:52:56+00:00
            if res["commit_date"] <= "2026-02-04T00:52:56+00:00":
                continue

            data.append(
                {
                    "Model": model_name,
                    "Checkpoint Index": i,
                    "Date": res["commit_date"],
                    "Pearson": res["pearson"] * 100,  # Convert to percentage
                    "Spearman": res["spearman"] * 100,
                }
            )

    return pd.DataFrame(data)


def plot_metric(df, metric, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure(figsize=(14, 7))

    # Create line plot
    sns.lineplot(
        data=df, x="Checkpoint Index", y=metric, hue="Model", marker="o", linewidth=2
    )

    plt.title(f"Version History - {metric}")
    plt.xlabel("Checkpoints (Ordered by Timestamp)")
    plt.ylabel(f"{metric} Score (%)")

    # Improve legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    # Adjust layout to prevent clipping of legend
    plt.tight_layout()

    # Save
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()


def main():
    print("Loading data...")
    df = load_data()

    if df.empty:
        print("No data found!")
        return

    print(f"Loaded {len(df)} data points.")

    print("Generating Pearson chart...")
    plot_metric(df, "Pearson", os.path.join("paper", "figures", "version_history_pearson.png"))

    print("Generating Spearman chart...")
    plot_metric(df, "Spearman", os.path.join("paper", "figures", "version_history_spearman.png"))

    print("Done.")


if __name__ == "__main__":
    main()
