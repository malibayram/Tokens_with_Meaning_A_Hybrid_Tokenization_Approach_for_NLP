#!/usr/bin/env python3
"""
Evaluate all committed versions of a HuggingFace model on Turkish STS benchmark.

Finds the best performing version based on Spearman correlation.

Usage:
    python evaluate_model_versions.py --model "alibayram/distilled-sentence-transformer-c400"
    python evaluate_model_versions.py --model "alibayram/distilled-sentence-transformer-c400" --split test
"""

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import list_repo_commits
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model_version(
    model_name: str,
    revision: str,
    split: str = "test",
    batch_size: int = 32,
    device: str | None = None,
    sentences1: list[str] = None,
    sentences2: list[str] = None,
    scores: list[float] = None,
) -> dict:
    """
    Evaluate a specific version (revision) of a model.

    Args:
        model_name: HuggingFace model ID
        revision: Git commit hash or branch name
        split: Dataset split to use
        batch_size: Batch size for encoding
        device: Device to use

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Loading model: {model_name} @ {revision[:8]}")

    try:
        model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {
            "revision": revision,
            "revision_short": revision[:8],
            "split": split,
            "error": str(e),
        }

    # Encode with timing
    all_sentences = sentences1 + sentences2
    start_time = time.time()
    embeddings = model.encode(
        all_sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    processing_time = time.time() - start_time

    # Split embeddings
    embeddings1 = embeddings[: len(sentences1)]
    embeddings2 = embeddings[len(sentences1) :]

    # Compute similarities
    similarities = (
        torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        .to(torch.float32)
        .cpu()
        .numpy()
    )

    # Compute correlations
    pearson = pearsonr(similarities, scores)[0]
    spearman = spearmanr(similarities, scores)[0]

    return {
        "revision": revision,
        "revision_short": revision[:8],
        "split": split,
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4),
        "num_samples": len(sentences1),
        "processing_time_seconds": round(processing_time, 2),
    }


def evaluate_all_versions(
    model_name: str,
    split: str = "test",
    batch_size: int = 32,
    device: str | None = None,
    output_file: str | None = None,
) -> list[dict]:
    """
    Evaluate all committed versions of a model.

    Args:
        model_name: HuggingFace model ID
        split: Dataset split to use
        batch_size: Batch size for encoding
        device: Device to use
        output_file: Path to save results JSON

    Returns:
        List of evaluation results for each version
    """
    # Get all commits for the model
    logger.info(f"Fetching commit history for {model_name}...")
    commits = list(list_repo_commits(model_name))

    logger.info(f"Found {len(commits)} versions to evaluate")

    results = []

    # Load dataset
    dataset = load_dataset("figenfikri/stsb_tr", split=split)

    sentences1 = list(dataset["sentence1"])
    sentences2 = list(dataset["sentence2"])
    scores = [float(s) / 5.0 for s in dataset["score"]]

    for i, commit in enumerate(commits):
        logger.info(
            f"\n[{i+1}/{len(commits)}] Evaluating revision: {commit.commit_id[:8]} ({commit.title})"
        )

        result = evaluate_model_version(
            model_name=model_name,
            revision=commit.commit_id,
            split=split,
            batch_size=batch_size,
            device=device,
            sentences1=sentences1,
            sentences2=sentences2,
            scores=scores,
        )

        # Add commit metadata
        result["commit_title"] = commit.title
        result["commit_date"] = (
            commit.created_at.isoformat() if commit.created_at else None
        )

        results.append(result)

        # Print intermediate result
        if "error" not in result:
            print(
                f"  Revision {result['revision_short']}: Pearson={result['pearson']:.4f}, Spearman={result['spearman']:.4f}"
            )

    # Sort by Spearman (best first)
    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda x: x["spearman"], reverse=True)

    # Print summary table
    print("\n" + "=" * 90)
    print(
        f"{'Revision':<10} {'Commit Date':<22} {'Pearson':>8} {'Spearman':>8} {'Time':>8}"
    )
    print("-" * 90)

    for r in valid_results:
        commit_date = r.get("commit_date", "")[:19] if r.get("commit_date") else "N/A"
        print(
            f"{r['revision_short']:<10} {commit_date:<22} {r['pearson']:>8.4f} {r['spearman']:>8.4f} {r['processing_time_seconds']:>6.2f}s"
        )

    print("=" * 90)

    # Identify best version
    if valid_results:
        best = valid_results[0]
        print(f"\n🏆 BEST VERSION: {best['revision_short']}")
        print(f"   Spearman: {best['spearman']:.4f}")
        print(f"   Pearson:  {best['pearson']:.4f}")
        print(f"   Date:     {best.get('commit_date', 'N/A')[:19]}")
        print(f"\n   To use this version:")
        print(
            f"   model = SentenceTransformer(\"{model_name}\", revision=\"{best['revision']}\")"
        )

    # Save results
    if output_file is None:
        # Generate filename from model name
        safe_name = model_name.replace("/", "_").replace("-", "_")
        output_file = f"version_eval_{safe_name}.json"

    output_data = {
        "model": model_name,
        "dataset": "figenfikri/stsb_tr",
        "split": split,
        "timestamp": datetime.now().isoformat(),
        "total_versions": len(commits),
        "successful_evaluations": len(valid_results),
        "best_version": valid_results[0] if valid_results else None,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all committed versions of a HuggingFace model on Turkish STS benchmark"
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="HuggingFace model ID (e.g., alibayram/distilled-sentence-transformer-c400)",
    )
    parser.add_argument(
        "--split",
        "-s",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=None,
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON file path (default: auto-generated)",
    )

    args = parser.parse_args()

    evaluate_all_versions(
        model_name=args.model,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
