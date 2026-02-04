#!/usr/bin/env python3
"""
Minimal evaluator for benchmarking models on Turkish STS benchmark.

Uses the figenfikri/stsb_tr dataset which contains Turkish sentence pairs
with human similarity scores (0-5 scale).

Usage:
    python evaluate_sts_tr.py --model "alibayram/distilled-sentence-transformer2"
    python evaluate_sts_tr.py --model "intfloat/multilingual-e5-small"

    # Compare multiple models
    python evaluate_sts_tr.py --model "model1" "model2" "model3"
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class STSResult:
    """Results from STS evaluation."""

    model_name: str
    split: str
    pearson: float
    spearman: float
    num_samples: int
    processing_time: float = 0.0  # Time in seconds

    def __str__(self) -> str:
        return (
            f"{self.model_name} on {self.split}:\n"
            f"  Pearson:  {self.pearson:.4f}\n"
            f"  Spearman: {self.spearman:.4f}\n"
            f"  Samples:  {self.num_samples}\n"
            f"  Time:     {self.processing_time:.2f}s"
        )


def evaluate_sts_tr(
    model: SentenceTransformer,
    split: str = "test",
    batch_size: int = 32,
    show_progress: bool = True,
) -> STSResult:
    """
    Evaluate a model on the Turkish STS benchmark.

    Args:
        model: SentenceTransformer model to evaluate
        split: Dataset split to use ('train', 'validation', 'test')
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar

    Returns:
        STSResult with Pearson and Spearman correlations
    """
    logger.info(f"Loading figenfikri/stsb_tr dataset (split: {split})")
    dataset = load_dataset("figenfikri/stsb_tr", split=split)

    # Extract sentence pairs and scores (convert to lists for compatibility)
    sentences1 = list(dataset["sentence1"])
    sentences2 = list(dataset["sentence2"])
    # Scores are strings in this dataset, convert to float and normalize to 0-1
    scores = [float(s) / 5.0 for s in dataset["score"]]

    logger.info(f"Encoding {len(sentences1)} sentence pairs...")

    # Encode all sentences with timing
    all_sentences = sentences1 + sentences2
    start_time = time.time()
    embeddings = model.encode(
        all_sentences,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_tensor=True,
    )
    processing_time = time.time() - start_time

    # Split embeddings back
    embeddings1 = embeddings[: len(sentences1)]
    embeddings2 = embeddings[len(sentences1) :]

    # Compute cosine similarities
    similarities = (
        torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        .float()
        .cpu()
        .numpy()
    )

    # Compute correlations
    pearson = pearsonr(similarities, scores)[0]
    spearman = spearmanr(similarities, scores)[0]

    # Get model name
    model_name = getattr(model, "_model_card_text", None)
    if hasattr(model, "model_card_data") and model.model_card_data:
        model_name = getattr(model.model_card_data, "model_name", None)
    if not model_name:
        model_name = str(model._first_module().__class__.__name__)

    return STSResult(
        model_name=model_name,
        split=split,
        pearson=pearson,
        spearman=spearman,
        num_samples=len(sentences1),
        processing_time=processing_time,
    )


def evaluate_model(
    model_name_or_path: str,
    splits: list[str] | None = None,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, STSResult]:
    """
    Evaluate a model on the Turkish STS benchmark.

    Args:
        model_name_or_path: HuggingFace model ID or path
        splits: Dataset splits to evaluate on
        batch_size: Batch size for encoding
        device: Device to use (auto-detected if None)

    Returns:
        Dictionary mapping split names to results
    """
    splits = splits or ["test"]

    logger.info(f"Loading model: {model_name_or_path}")
    model = SentenceTransformer(
        model_name_or_path, device=device, trust_remote_code=True
    )

    results = {}
    for split in splits:
        result = evaluate_sts_tr(model, split=split, batch_size=batch_size)
        result.model_name = model_name_or_path
        results[split] = result
        print(f"\n{result}")

    return results


def save_results_to_json(
    results: list,
    output_file: str = "sts_benchmark_results.json",
) -> None:
    """
    Save benchmark results to a JSON file with timestamp.

    Args:
        results: List of STSResult objects
        output_file: Path to the JSON output file
    """
    # Prepare the new entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "figenfikri/stsb_tr",
        "results": [
            {
                "model": result.model_name,
                "pearson": round(result.pearson, 4),
                "spearman": round(result.spearman, 4),
                "num_samples": result.num_samples,
                "split": result.split,
                "processing_time_seconds": round(result.processing_time, 2),
            }
            for result in results
        ],
    }

    # Load existing data or create new list
    output_path = Path(output_file)
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append new entry
    data.append(entry)

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_file}")


def compare_models(
    model_names: list[str],
    split: str = "test",
    batch_size: int = 32,
    device: str | None = None,
) -> None:
    """
    Compare multiple models on the Turkish STS benchmark.

    Args:
        model_names: List of HuggingFace model IDs or paths
        split: Dataset split to evaluate on
        batch_size: Batch size for encoding
        device: Device to use
    """
    print("\n" + "=" * 70)
    print("Turkish STS Benchmark Comparison")
    print("Dataset: figenfikri/stsb_tr")
    print("=" * 70)

    results = []
    for model_name in model_names:
        try:
            result = evaluate_model(
                model_name,
                splits=[split],
                batch_size=batch_size,
                device=device,
            )[split]
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Model':<50} {'Pearson':>8} {'Spearman':>8} {'Time':>10}")
    print("-" * 80)

    # Sort by Spearman correlation (higher is better)
    results.sort(key=lambda x: x.spearman, reverse=True)

    for result in results:
        model_display = (
            result.model_name[:48] + ".."
            if len(result.model_name) > 50
            else result.model_name
        )
        print(
            f"{model_display:<50} {result.pearson:>8.4f} {result.spearman:>8.4f} {result.processing_time:>8.2f}s"
        )

    print("=" * 80)

    # Save results to JSON file
    if results:
        save_results_to_json(results)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on Turkish STS benchmark (figenfikri/stsb_tr)"
    )
    parser.add_argument(
        "--model",
        "-m",
        nargs="+",
        required=True,
        help="Model name(s) or path(s) to evaluate",
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
        "--all-splits",
        action="store_true",
        help="Evaluate on all splits (train, validation, test)",
    )

    args = parser.parse_args()

    if len(args.model) == 1:
        # Single model evaluation
        splits = ["train", "validation", "test"] if args.all_splits else [args.split]
        results = evaluate_model(
            args.model[0],
            splits=splits,
            batch_size=args.batch_size,
            device=args.device,
        )
        # Save results to JSON
        save_results_to_json(list(results.values()))
    else:
        # Multiple model comparison
        compare_models(
            args.model,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()
