"""Minimal Embedding Distillation Trainer using precomputed teacher embeddings."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import get_scheduler

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingTrainerConfig:
    """Configuration for embedding distillation training."""

    # Model
    student_model: str

    # Training
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Limit training steps (optional, for debugging/warmup)
    max_steps: int | None = None

    # Loss
    loss_type: Literal["mse", "cosine"] = "mse"

    # Columns (use pre-encoded input_ids from prepare_dataset.py)
    input_ids_column: str = "input_ids"
    embedding_column: str = "teacher_embedding_final"

    # Output
    output_dir: str = "./trained_model"
    save_steps: int = 1000
    logging_steps: int = 100

    # Hub
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_token: str | None = None

    # Device
    device: str = "auto"

    # Optimization
    use_fp16: bool = False
    use_bf16: bool = False
    compile_model: bool = False
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False

    # WandB
    use_wandb: bool = False
    wandb_project: str = "distillation"
    wandb_run_name: str | None = None


class EmbeddingDistillationTrainer:
    """
    Minimal trainer for distillation using precomputed teacher embeddings.

    Example:
        >>> config = EmbeddingTrainerConfig(
        ...     student_model="sentence-transformers/all-MiniLM-L6-v2",
        ...     use_bf16=True,
        ...     compile_model=True,
        ... )
        >>> trainer = EmbeddingDistillationTrainer(config)
        >>> trainer.train("your-username/corpus-with-embeddings")
    """

    def __init__(self, config: EmbeddingTrainerConfig):
        self.config = config
        self.device = self._get_device()
        self._original_model = None
        self.student_model = self._load_student()
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0

    def _get_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.config.device)

    def _load_student(self) -> SentenceTransformer:
        logger.info(f"Loading student: {self.config.student_model}")

        model_kwargs = {}
        if self.config.use_flash_attention:
            try:
                import flash_attn

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Enabling Flash Attention 2")
            except ImportError:
                logger.warning("flash-attn not installed, skipping Flash Attention")

        model_kwargs["trust_remote_code"] = True

        model = SentenceTransformer(
            self.config.student_model,
            model_kwargs=model_kwargs,
            trust_remote_code=True,
        )
        model.to(self.device)

        if self.config.use_bf16 and self.device.type == "cuda":
            model = model.to(dtype=torch.bfloat16)
            logger.info("Using bfloat16 precision")
        elif self.config.use_fp16:
            model = model.half()
            logger.info("Using float16 precision")

        if self.config.gradient_checkpointing:
            for module in model.modules():
                if hasattr(module, "gradient_checkpointing_enable"):
                    module.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        self._original_model = model

        if self.config.compile_model:
            try:
                compiled = torch.compile(model)
                logger.info("Model compiled with torch.compile()")
                return compiled
            except Exception as e:
                logger.warning(f"torch.compile() failed: {e}")

        return model

    def _get_loss_fn(self):
        if self.config.loss_type == "mse":
            return torch.nn.MSELoss()
        elif self.config.loss_type == "cosine":
            return torch.nn.CosineEmbeddingLoss()
        raise ValueError(f"Unknown loss: {self.config.loss_type}")

    def train(self, dataset: str | Dataset, split: str = "train") -> dict:
        """Train the student model."""
        if isinstance(dataset, str):
            logger.info(f"Loading dataset: {dataset}")
            train_ds = load_dataset(dataset, split=split)
        else:
            train_ds = dataset

        input_ids_col = self.config.input_ids_column
        embedding_col = self.config.embedding_column

        logger.info(f"Using input_ids column: {input_ids_col}")
        logger.info(f"Dataset size: {len(train_ds)}")

        # Initialize WandB
        if self.config.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config={
                        "student_model": self.config.student_model,
                        "learning_rate": self.config.learning_rate,
                        "num_epochs": self.config.num_epochs,
                        "batch_size": self.config.batch_size,
                        "loss_type": self.config.loss_type,
                    },
                )
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, disabling wandb logging")
                self._wandb = None
        else:
            self._wandb = None

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Setup scheduler
        total_steps = (len(train_ds) // self.config.batch_size) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        loss_fn = self._get_loss_fn()
        self.student_model.train()
        self.global_step = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            indices = list(range(len(train_ds)))

            progress = tqdm(
                range(0, len(train_ds), self.config.batch_size),
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            )

            for start_idx in progress:
                end_idx = min(start_idx + self.config.batch_size, len(train_ds))
                batch = train_ds.select(indices[start_idx:end_idx])

                # Get pre-encoded input_ids and targets
                input_ids_list = batch[input_ids_col]
                targets = torch.tensor(batch[embedding_col]).to(self.device)

                # Pad input_ids to same length
                max_len = max(len(ids) for ids in input_ids_list)
                # Hardcode pad_id to 0 as requested/safe default
                pad_id = 0
                padded = [
                    ids + [pad_id] * (max_len - len(ids)) for ids in input_ids_list
                ]
                features = {"input_ids": torch.tensor(padded).to(self.device)}

                # Forward pass
                student_output = self._forward(features)
                targets = targets.to(dtype=student_output.dtype)

                # Compute loss
                if self.config.loss_type == "cosine":
                    labels = torch.ones(
                        len(input_ids_list), dtype=student_output.dtype
                    ).to(self.device)
                    loss = loss_fn(student_output, targets, labels)
                else:
                    loss = loss_fn(student_output, targets)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(), self.config.max_grad_norm
                    )

                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1

                progress.set_postfix({"loss": f"{loss.item():.4f}"})

                if (
                    self.global_step <= 5
                    or self.global_step % self.config.logging_steps == 0
                ):
                    avg_loss = epoch_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {self.global_step}: loss = {avg_loss:.4f}, lr = {lr:.2e}"
                    )

                    if self._wandb:
                        self._wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": lr,
                                "train/epoch": epoch + 1,
                            },
                            step=self.global_step,
                        )

                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break

                if self.global_step % self.config.save_steps == 0:
                    """self.save_model(
                        f"{self.config.output_dir}/checkpoint-{self.global_step}"
                    )"""
                    if self.config.push_to_hub and self.config.hub_model_id:
                        logger.info(f"Pushing checkpoint {self.global_step} to hub...")
                        self.push_to_hub()

            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1}: avg_loss = {avg_epoch_loss:.4f}")

        self.save_model(self.config.output_dir)

        if self.config.push_to_hub and self.config.hub_model_id:
            self.push_to_hub()

        if self._wandb:
            self._wandb.finish()

        return {"train_loss": avg_epoch_loss}

    def _forward(self, features: dict) -> torch.Tensor:
        """Forward pass through full model."""
        model = self._original_model

        for module in model:
            features = module(features)

        embedding = features.get("sentence_embedding")
        if embedding is None:
            embedding = features.get("token_embeddings")
            if embedding is not None and len(embedding.shape) == 3:
                embedding = embedding.mean(dim=1)

        return embedding

    def save_model(self, path: str) -> None:
        """Save the student model."""
        self._original_model.save(path)
        logger.info(f"Model saved: {path}")

    def push_to_hub(self) -> str | None:
        """Push model to HuggingFace Hub with retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = self._original_model.push_to_hub(
                    repo_id=self.config.hub_model_id,
                    token=self.config.hub_token,
                    exist_ok=True,
                )
                logger.info(f"Pushed to: {url}")
                return url
            except Exception as e:
                logger.warning(
                    f"Failed to push to hub (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    logger.info("Retrying in 30 seconds...")
                    time.sleep(30)
                else:
                    logger.error(
                        "All attempts to push to hub failed. Continuing pipeline..."
                    )
                    return None
