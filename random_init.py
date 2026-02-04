"""
Random Initialization Script for MFT, tabiBERT, cosmosGPT2 and newmindaiMursit Tokenizers

Creates a single SentenceTransformer model with random initialization
and pushes it to MFT, tabiBERT, cosmosGPT2 and newmindaiMursit repositories.
Since all use identical random weights (seed=42), we only need to create once.

"""

import os
import shutil
import torch
import torch.nn as nn
import random
import numpy as np

from dotenv import load_dotenv
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# Fixed seed for reproducibility
SEED = 42
VOCAB_SIZE = 32768  # Both tokenizers have same vocab size
org_model_id = "google/embeddinggemma-300m"
clone_dir = "random_init_cloned"


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(module):
    """Initialize all weights randomly with Xavier/Glorot initialization."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "weight") and module.weight is not None:
        if module.weight.dim() >= 2:
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


if __name__ == "__main__":
    print(f"Random Initialization with SEED={SEED}")
    print("Creating single model, pushing to all repos")
    print("=" * 60)

    # Set seed
    set_seed(SEED)

    # Load SentenceTransformer structure
    print("Loading SentenceTransformer structure...")
    model = SentenceTransformer(org_model_id, token=HF_TOKEN)

    # Resize embeddings to 32K
    model[0].auto_model.resize_token_embeddings(VOCAB_SIZE)

    # Apply random initialization
    set_seed(SEED)
    print(f"Applying random initialization with seed={SEED}...")
    model.apply(init_weights)
    model = model.to(torch.bfloat16)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Save model
    model.save_pretrained(clone_dir)

    # push model to hub
    print("\n--- Pushing model to hub ---")
    model.push_to_hub("alibayram/downstream-random-init", token=HF_TOKEN, exist_ok=True)
    print("✓ Uploaded alibayram/downstream-random-init")

    # === Push to MFT repo (with tokenizer) ===
    print("\n--- MFT Repository ---")

    # Remove tokenizer files
    for f in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
    ]:
        path = os.path.join(clone_dir, f)
        if os.path.exists(path):
            os.remove(path)

    # Load with custom tokenizer and push
    mft_tokenizer = AutoTokenizer.from_pretrained(
        "alibayram/turkish-mft-tokenizer", trust_remote_code=True, use_fast=False
    )
    print(f"MFT tokenizer vocab size: {mft_tokenizer.vocab_size}")
    mft_tokenizer.save_pretrained(clone_dir)

    mft_model = SentenceTransformer(clone_dir, trust_remote_code=True)
    mft_model = mft_model.to(torch.bfloat16)

    print("Uploading to alibayram/mft-random...")
    mft_model.push_to_hub("alibayram/mft-random", token=HF_TOKEN, exist_ok=True)
    print("✓ Uploaded alibayram/mft-random")

    del mft_model

    # === Push to TabiBERT repo (with tokenizer) ===
    print("\n--- TabiBERT Repository ---")

    # Add TabiBERT tokenizer
    tabi_tokenizer = AutoTokenizer.from_pretrained(
        "alibayram/tabi-random-init", use_fast=False
    )
    print(f"TabiBERT tokenizer vocab size: {tabi_tokenizer.vocab_size}")
    tabi_tokenizer.save_pretrained(clone_dir)

    # Reload and push
    tabi_model = SentenceTransformer(clone_dir)
    tabi_model = tabi_model.to(torch.bfloat16)

    print("Uploading to alibayram/tabi-random...")
    tabi_model.push_to_hub("alibayram/tabi-random", token=HF_TOKEN, exist_ok=True)
    print("✓ Uploaded alibayram/tabi-random")

    del tabi_model

    # === Push to cosmosGPT2-random-init ===
    print("\n--- cosmosGPT2-random-init Repository ---")

    cosmos_tokenizer = AutoTokenizer.from_pretrained(
        "alibayram/cosmosGPT2-random-init", use_fast=False
    )
    print(f"cosmosGPT2 tokenizer vocab size: {cosmos_tokenizer.vocab_size}")
    cosmos_tokenizer.save_pretrained(clone_dir)

    # Reload and push
    cosmos_model = SentenceTransformer(clone_dir)
    cosmos_model = cosmos_model.to(torch.bfloat16)

    print("Uploading to alibayram/cosmosGPT2-random...")
    cosmos_model.push_to_hub(
        "alibayram/cosmosGPT2-random", token=HF_TOKEN, exist_ok=True
    )
    print("✓ Uploaded alibayram/cosmosGPT2-random")

    del cosmos_model

    # === Push to newmindaiMursit-random-init ===
    print("\n--- newmindaiMursit-random-init Repository ---")

    newmindai_tokenizer = AutoTokenizer.from_pretrained(
        "alibayram/newmindaiMursit-random-init", use_fast=False
    )
    print(f"newmindaiMursit tokenizer vocab size: {newmindai_tokenizer.vocab_size}")
    newmindai_tokenizer.save_pretrained(clone_dir)

    # Reload and push
    newmindai_model = SentenceTransformer(clone_dir)
    newmindai_model = newmindai_model.to(torch.bfloat16)

    print("Uploading to alibayram/newmindaiMursit-random...")
    newmindai_model.push_to_hub(
        "alibayram/newmindaiMursit-random", token=HF_TOKEN, exist_ok=True
    )
    print("✓ Uploaded alibayram/newmindaiMursit-random")

    # Cleanup
    shutil.rmtree(clone_dir)

    print("\n" + "=" * 60)
    print("✓ Both repos updated with identical random weights!")
    print(f"✓ Seed: {SEED}")
    print(f"✓ Vocab size: {VOCAB_SIZE}")
    print("=" * 60)
