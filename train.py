"""Training script for embedding distillation using EmbeddingDistillationTrainer.

This module configures and runs the embedding distillation process,
training a student model to match teacher embeddings from a pre-encoded dataset.

Prerequisites:
    1. Run prepare_dataset.py first to create the encoded dataset
    2. Create .env file with: WANDB_API_KEY=xxx and HF_TOKEN=xxx
"""

import logging
import os
import sys
import traceback

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

from embedding_trainer import EmbeddingDistillationTrainer, EmbeddingTrainerConfig

load_dotenv()

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    logger.warning("HF_TOKEN not found in environment variables or .env file!")
if not WANDB_API_KEY:
    logger.warning("WANDB_API_KEY not found in environment variables or .env file!")

if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

# Pre-encoded dataset from prepare_dataset.py
DATASET_ID = "alibayram/cosmos-corpus-encoded"

# Models configuration (Just the 2 new random models)
MODELS = [
    {
        "name": "mft-random",
        "model_id": "alibayram/mft-random",
        "input_ids_column": "mft_input_ids",
    },
    {
        "name": "tabi-random",
        "model_id": "alibayram/tabi-random",
        "input_ids_column": "tabi_input_ids",
    },
    {
        "name": "cosmos-random",
        "model_id": "alibayram/cosmosGPT2-random",
        "input_ids_column": "cosmos_input_ids",
    },
    {
        "name": "mursit-random",
        "model_id": "alibayram/newmindaiMursit-random",
        "input_ids_column": "mursit_input_ids",
    },
]

logger.info(f"Found {len(MODELS)} models to train.")

# --- Phase 1: Warmup (100 steps) ---
logger.info("--- Phase 1: Warmup (100 steps) ---")

for i, model_cfg in enumerate(MODELS):
    model_name = model_cfg["name"]
    model_id = model_cfg["model_id"]
    input_column = model_cfg["input_ids_column"]

    logger.info(f"\n[{i+1}/{len(MODELS)}] Starting training for: {model_name}")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Input Column: {input_column}")

    warmup_output_dir = f"./trained_models/{model_name}_warmup"

    warmup_config = EmbeddingTrainerConfig(
        student_model=model_id,
        num_epochs=1,  # Will be limited by max_stepsx
        max_steps=100,
        batch_size=256 + 128,
        learning_rate=5e-5,
        warmup_ratio=0.01,
        weight_decay=0.01,
        loss_type="cosine",
        input_ids_column=input_column,
        embedding_column="teacher_embedding_final",
        use_bf16=True,
        gradient_checkpointing=True,
        compile_model=True,
        output_dir=warmup_output_dir,
        save_steps=25,
        logging_steps=5,
        use_wandb=True,
        wandb_project="mft-downstream",
        wandb_run_name=f"{model_name}-warmup",
        push_to_hub=True,
        hub_model_id=model_id,
        hub_token=HF_TOKEN,
    )

    try:
        trainer = EmbeddingDistillationTrainer(warmup_config)
        metrics = trainer.train(DATASET_ID)
        logger.info(
            f"✓ Finished Warmup {model_name}. Loss: {metrics['train_loss']:.4f}"
        )
        del trainer
        del warmup_config
    except Exception:
        logger.error(f"✗ Failed Warmup {model_name}")
        traceback.print_exc()
        continue

    # Cleanup
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Phase 2: Full Training (1 Epoch) ---
logger.info("--- Phase 2: Full Training (1 Epoch) ---")

for i, model_cfg in enumerate(MODELS):
    continue
    model_name = model_cfg["name"]
    model_id = model_cfg["model_id"]
    input_column = model_cfg["input_ids_column"]

    logger.info(f"\n[{i+1}/{len(MODELS)}] Starting training for: {model_name}")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Input Column: {input_column}")

    full_config = EmbeddingTrainerConfig(
        student_model=model_id,
        num_epochs=1,
        max_steps=None,  # Run full epoch if None
        batch_size=256 + 128,
        learning_rate=5e-5,
        warmup_ratio=0.01,
        weight_decay=0.01,
        loss_type="cosine",
        input_ids_column=input_column,
        embedding_column="teacher_embedding_final",
        use_bf16=True,
        gradient_checkpointing=True,
        compile_model=True,
        output_dir=f"./trained_models/{model_name}",
        save_steps=25,  # Only save at end (handled by trainer)
        logging_steps=5,
        use_wandb=True,
        wandb_project="mft-downstream",
        wandb_run_name=f"{model_name}-full",
        push_to_hub=True,
        hub_model_id=model_id,
        hub_token=HF_TOKEN,
    )

    try:
        trainer = EmbeddingDistillationTrainer(full_config)
        metrics = trainer.train(DATASET_ID)
        logger.info(
            f"✓ Finished Full Training {model_name}. Loss: {metrics['train_loss']:.4f}"
        )
    except Exception:
        logger.error(f"✗ Failed Full Training {model_name}")
        traceback.print_exc()

    # Cleanup memory
    if "trainer" in locals():
        del trainer
    if "full_config" in locals():
        del full_config

    # Cleanup
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleared CUDA cache for next model.\n")
