#!/usr/bin/env python3
"""
TurBLiMP evaluation for embedding models via a centroid-based proxy.

For each phenomenon CSV (good_sentence, bad_sentence):
  1) Embed all good+bad sentences (normalized embeddings)
  2) Compute centroid over good sentence embeddings (then normalize centroid)
  3) Score(sentence) = cosine(sentence, centroid) = embedding dot centroid
  4) Count a pair correct iff score(good) > score(bad)
  5) Report accuracy (%) per phenomenon

Outputs (by default into ./turblimp_results_tables):
  - results_table_centroid.csv / .html / .tex
  - pairwise_details/<model>/<phenomenon>.csv  (optional)
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_pairs(filepath: Path, delimiter: str = ";") -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with filepath.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        _header = next(reader, None)
        for row in reader:
            if not row or len(row) < 2:
                continue
            good = (row[0] or "").strip()
            bad = (row[1] or "").strip()
            if good and bad:
                pairs.append((good, bad))
    return pairs


def encode(
    model: SentenceTransformer, sentences: list[str], batch_size: int
) -> np.ndarray:
    return model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def score_centroid(
    E_good: np.ndarray, E_bad: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    centroid = E_good.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    good_scores = E_good @ centroid
    bad_scores = E_bad @ centroid
    return good_scores, bad_scores


def evaluate_file(
    model: SentenceTransformer,
    filepath: Path,
    delimiter: str,
    batch_size: int,
) -> tuple[float, float, pd.DataFrame]:
    pairs = load_pairs(filepath, delimiter=delimiter)
    if not pairs:
        return float("nan"), float("nan"), pd.DataFrame()

    good_sents = [g for g, _b in pairs]
    bad_sents = [b for _g, b in pairs]
    E = encode(model, good_sents + bad_sents, batch_size=batch_size)
    E_good = E[: len(good_sents)]
    E_bad = E[len(good_sents) :]

    good_scores, bad_scores = score_centroid(E_good, E_bad)
    diffs = good_scores - bad_scores
    correct = diffs > 0
    acc = float(correct.mean()) * 100.0
    mean_diff = float(diffs.mean())

    details = pd.DataFrame(
        {
            "good_sentence": good_sents,
            "bad_sentence": bad_sents,
            "good_score": good_scores.astype(float),
            "bad_score": bad_scores.astype(float),
            "difference": diffs.astype(float),
            "correct": correct.astype(bool),
        }
    )
    return acc, mean_diff, details


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate embedding models on TurBLiMP (centroid proxy)."
    )
    ap.add_argument(
        "--data_dir",
        default="TurBLiMP/data",
        help="TurBLiMP data directory containing experimental/*.csv (default: TurBLiMP/data)",
    )
    ap.add_argument(
        "--experimental_subdir",
        default="experimental",
        help="Subdirectory under data_dir with phenomenon CSVs (default: experimental)",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more SentenceTransformer model names/paths to evaluate.",
    )
    ap.add_argument("--delimiter", default=";", help="CSV delimiter (default: ';').")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--device", default=None, help="Device for SentenceTransformer (default: auto)."
    )
    ap.add_argument(
        "--output_dir",
        default="turblimp_results_tables",
        help="Output directory for summary tables (default: turblimp_results_tables).",
    )
    ap.add_argument(
        "--write_pairwise_details",
        action="store_true",
        help="Write per-pair CSVs under <output_dir>/pairwise_details/<model>/",
    )
    args = ap.parse_args()

    exp_dir = Path(args.data_dir) / args.experimental_subdir
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experimental folder not found: {exp_dir}")

    csv_files = sorted(exp_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {exp_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [fp.stem for fp in csv_files]
    table = pd.DataFrame(index=rows)

    for model_name in args.models:
        model = SentenceTransformer(
            model_name, device=args.device, trust_remote_code=True
        )
        col = (
            model_name.split("/")[-1].replace("-random", "").replace("-random-init", "")
        )

        scores = []
        for fp in csv_files:
            acc, mean_diff, details = evaluate_file(
                model=model,
                filepath=fp,
                delimiter=args.delimiter,
                batch_size=args.batch_size,
            )
            scores.append(acc)

            if args.write_pairwise_details:
                safe_model = model_name.replace("/", "__")
                det_dir = out_dir / "pairwise_details" / safe_model
                det_dir.mkdir(parents=True, exist_ok=True)
                details.to_csv(
                    det_dir / f"{fp.stem}.csv", index=False, encoding="utf-8"
                )

        table[col] = scores

    table.loc["Model Average"] = table.mean(axis=0)

    # Round to 1 decimal for presentation
    table_rounded = table.round(1)

    csv_path = out_dir / "results_table_centroid.csv"
    table_rounded.to_csv(csv_path, encoding="utf-8")

    # HTML with colors (source of truth for cell colors used in the paper)
    styler = table_rounded.style.background_gradient(cmap="RdYlGn")
    html_path = out_dir / "results_table_centroid.html"
    styler.to_html(html_path)

    # Plain LaTeX (non-colored; paper uses the colored TeX generated from HTML)
    tex_path = out_dir / "results_table_centroid.tex"
    tex = table_rounded.to_latex(index=True, escape=True, float_format="%.1f")
    tex_path.write_text(tex, encoding="utf-8")

    print(f"✓ Wrote {csv_path}")
    print(f"✓ Wrote {html_path}")
    print(f"✓ Wrote {tex_path}")
    if args.write_pairwise_details:
        print(f"✓ Wrote pairwise details under {out_dir / 'pairwise_details'}")


if __name__ == "__main__":
    main()
