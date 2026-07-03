"""
Training loop for TinyStories language models.

Implements the full training pipeline with:
- AdamW optimizer with weight decay
- Cosine learning rate schedule with linear warmup
- Mixed precision (AMP) training for efficiency
- Gradient accumulation for effective larger batch sizes
- Periodic checkpointing and validation
- Optional Weights & Biases logging

The training follows standard next-token prediction: given a sequence of
tokens [t1, t2, ..., tn], the model predicts [t2, t3, ..., tn+1] using
cross-entropy loss.

Usage:
    python -m tinystories.train \
        --config tiny-3M \
        --data_dir data/tinystories \
        --output_dir checkpoints/tiny-3M \
        --epochs 10

Reference: Section 3 of Eldan & Li (2023), arXiv:2305.07759
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tinystories.config import (
    ModelConfig,
    TrainConfig,
    get_default_train_config,
    get_model_config,
)
from tinystories.data.prepare import TinyStoriesDataset
from tinystories.model.transformer import TinyStoriesModel

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """Determine the best available device.

    Args:
        preference: One of 'auto', 'cuda', 'mps', 'cpu'.

    Returns:
        torch.device for training.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preference)


def get_lr(step: int, train_config: TrainConfig, total_steps: int) -> float:
    """Compute learning rate with linear warmup and cosine decay.

    Schedule:
    1. Linear warmup from 0 to peak LR over warmup_steps
    2. Cosine decay from peak LR to min_lr over remaining steps

    Args:
        step: Current training step.
        train_config: Training configuration.
        total_steps: Total number of training steps.

    Returns:
        Learning rate for the current step.
    """
    peak_lr = train_config.learning_rate
    min_lr = peak_lr * train_config.min_lr_ratio

    # Linear warmup
    if step < train_config.warmup_steps:
        return peak_lr * (step + 1) / train_config.warmup_steps

    # Cosine decay
    if train_config.lr_scheduler == "cosine":
        decay_steps = total_steps - train_config.warmup_steps
        progress = (step - train_config.warmup_steps) / max(1, decay_steps)
        progress = min(progress, 1.0)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    # Linear decay fallback
    decay_ratio = (step - train_config.warmup_steps) / max(
        1, total_steps - train_config.warmup_steps
    )
    return peak_lr - (peak_lr - min_lr) * decay_ratio


@torch.no_grad()
def estimate_loss(
    model: TinyStoriesModel,
    dataset: TinyStoriesDataset,
    device: torch.device,
    eval_steps: int = 50,
    batch_size: int = 32,
) -> float:
    """Estimate validation loss over a number of batches.

    Args:
        model: The language model to evaluate.
        dataset: Validation dataset.
        device: Device for computation.
        eval_steps: Number of evaluation batches.
        batch_size: Batch size for evaluation.

    Returns:
        Average cross-entropy loss.
    """
    model.eval()
    losses = []

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    for i, (x, y) in enumerate(loader):
        if i >= eval_steps:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return np.mean(losses) if losses else float("inf")


def save_checkpoint(
    model: TinyStoriesModel,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    step: int,
    epoch: int,
    val_loss: float,
    output_dir: str,
    is_best: bool = False,
):
    """Save a training checkpoint.

    Saves model weights, optimizer state, and metadata needed to
    resume training or use the model for inference.

    Args:
        model: The model to save.
        optimizer: Optimizer state.
        model_config: Architecture configuration.
        train_config: Training configuration.
        step: Current global step.
        epoch: Current epoch.
        val_loss: Latest validation loss.
        output_dir: Directory to save the checkpoint.
        is_best: If True, also save as 'best.pt'.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": vars(model_config),
        "train_config": vars(train_config),
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss,
    }

    # Save numbered checkpoint
    ckpt_path = out_path / f"checkpoint_step{step}.pt"
    torch.save(checkpoint, str(ckpt_path))
    logger.info(f"Saved checkpoint to {ckpt_path}")

    # Save as 'best.pt' if it's the best so far
    if is_best:
        best_path = out_path / "best.pt"
        torch.save(checkpoint, str(best_path))
        logger.info(f"New best model! val_loss={val_loss:.4f} -> {best_path}")

    # Save latest (for easy resumption)
    latest_path = out_path / "latest.pt"
    torch.save(checkpoint, str(latest_path))


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    """Load a training checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to map tensors to.

    Returns:
        Checkpoint dictionary.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info(
        f"Loaded checkpoint from {checkpoint_path} "
        f"(step={checkpoint.get('step', '?')}, "
        f"val_loss={checkpoint.get('val_loss', '?'):.4f})"
    )
    return checkpoint


def train(
    model_config: ModelConfig,
    train_config: TrainConfig,
    resume_from: str = None,
):
    """Main training function.

    Sets up the model, optimizer, data loaders, and runs the training loop
    with periodic validation and checkpointing.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameters.
        resume_from: Optional checkpoint path to resume training from.
    """
    device = get_device(train_config.device)
    logger.info(f"Training on device: {device}")

    # Set random seed
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    # Set up mixed precision
    use_amp = train_config.use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if train_config.dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    logger.info(f"Mixed precision: {use_amp} (dtype={train_config.dtype})")

    # Load datasets
    logger.info(f"Loading training data from {train_config.data_dir}")
    train_dataset = TinyStoriesDataset(
        train_config.data_dir,
        split="train",
        context_length=model_config.context_length,
    )

    val_dataset = None
    try:
        val_dataset = TinyStoriesDataset(
            train_config.data_dir,
            split="validation",
            context_length=model_config.context_length,
        )
    except FileNotFoundError:
        logger.warning("No validation split found. Skipping validation.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # Create model
    model = TinyStoriesModel(model_config).to(device)

    # Optimizer: AdamW with weight decay applied only to weight matrices
    # (not biases or LayerNorm parameters) following GPT-2 conventions
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
    )

    # Resume from checkpoint if specified
    start_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    if resume_from and os.path.exists(resume_from):
        checkpoint = load_checkpoint(resume_from, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint.get("step", 0)
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        logger.info(f"Resumed from step {start_step}, epoch {start_epoch}")

    # Calculate total steps
    steps_per_epoch = len(train_loader) // train_config.gradient_accumulation_steps
    total_steps = steps_per_epoch * train_config.max_epochs
    if train_config.max_steps:
        total_steps = train_config.max_steps

    logger.info(
        f"Training for {train_config.max_epochs} epochs "
        f"({total_steps} steps, {steps_per_epoch} steps/epoch)"
    )

    # Optional W&B logging
    wandb_run = None
    if train_config.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=train_config.wandb_project,
                name=train_config.wandb_run_name,
                config={
                    "model": vars(model_config),
                    "train": vars(train_config),
                },
            )
        except ImportError:
            logger.warning("wandb not installed. Skipping W&B logging.")

    # Training loop
    model.train()
    global_step = start_step
    optimizer.zero_grad()

    for epoch in range(start_epoch, train_config.max_epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for micro_step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward pass with optional AMP
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    _, loss = model(x, y)
                    loss = loss / train_config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                _, loss = model(x, y)
                loss = loss / train_config.gradient_accumulation_steps
                loss.backward()

            epoch_loss += loss.item() * train_config.gradient_accumulation_steps
            epoch_tokens += x.numel()

            # Gradient accumulation step
            if (micro_step + 1) % train_config.gradient_accumulation_steps == 0:
                # Update learning rate
                lr = get_lr(global_step, train_config, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # Gradient clipping
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.max_grad_norm
                )

                # Optimizer step
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % train_config.log_every_steps == 0:
                    dt = time.time() - t0
                    tokens_per_sec = epoch_tokens / dt
                    avg_loss = epoch_loss / (micro_step + 1)
                    logger.info(
                        f"step {global_step}/{total_steps} | "
                        f"epoch {epoch+1}/{train_config.max_epochs} | "
                        f"loss {avg_loss:.4f} | lr {lr:.2e} | "
                        f"{tokens_per_sec:.0f} tok/s"
                    )

                    if wandb_run:
                        wandb_run.log({
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/epoch": epoch,
                        }, step=global_step)

                # Validation
                if (
                    val_dataset
                    and global_step % train_config.eval_every_steps == 0
                ):
                    val_loss = estimate_loss(
                        model, val_dataset, device,
                        batch_size=train_config.batch_size,
                    )
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss

                    logger.info(
                        f"Validation loss: {val_loss:.4f} "
                        f"(best: {best_val_loss:.4f})"
                    )

                    if wandb_run:
                        wandb_run.log({
                            "val/loss": val_loss,
                            "val/best_loss": best_val_loss,
                        }, step=global_step)

                    save_checkpoint(
                        model, optimizer, model_config, train_config,
                        global_step, epoch, val_loss,
                        train_config.output_dir, is_best=is_best,
                    )
                    model.train()

                # Periodic checkpoint (without validation)
                elif global_step % train_config.save_every_steps == 0:
                    save_checkpoint(
                        model, optimizer, model_config, train_config,
                        global_step, epoch, best_val_loss,
                        train_config.output_dir,
                    )

                # Check max steps
                if train_config.max_steps and global_step >= train_config.max_steps:
                    logger.info(f"Reached max_steps={train_config.max_steps}. Stopping.")
                    break

        # End of epoch
        dt = time.time() - t0
        logger.info(
            f"Epoch {epoch + 1} complete in {dt:.1f}s. "
            f"Avg loss: {epoch_loss / max(1, micro_step + 1):.4f}"
        )

        if train_config.max_steps and global_step >= train_config.max_steps:
            break

    # Final save
    final_val_loss = best_val_loss
    if val_dataset:
        final_val_loss = estimate_loss(
            model, val_dataset, device, batch_size=train_config.batch_size
        )
    save_checkpoint(
        model, optimizer, model_config, train_config,
        global_step, train_config.max_epochs, final_val_loss,
        train_config.output_dir,
        is_best=final_val_loss <= best_val_loss,
    )

    if wandb_run:
        wandb_run.finish()

    logger.info("Training complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train a TinyStories language model")
    parser.add_argument(
        "--config", type=str, default="tiny-3M",
        help="Model configuration preset (default: tiny-3M)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/tinystories",
        help="Directory containing tokenized training data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--save_every", type=int, default=None, help="Save every N steps")
    parser.add_argument("--eval_every", type=int, default=None, help="Evaluate every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cpu/mps")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configurations
    model_config = get_model_config(args.config)
    train_config = get_default_train_config(args.config)

    # Apply CLI overrides
    train_config.data_dir = args.data_dir
    train_config.output_dir = args.output_dir
    train_config.seed = args.seed
    train_config.device = args.device

    if args.epochs:
        train_config.max_epochs = args.epochs
    if args.max_steps:
        train_config.max_steps = args.max_steps
    if args.lr:
        train_config.learning_rate = args.lr
    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.save_every:
        train_config.save_every_steps = args.save_every
    if args.eval_every:
        train_config.eval_every_steps = args.eval_every
    if args.wandb_project:
        train_config.wandb_project = args.wandb_project

    logger.info(f"Model config: {vars(model_config)}")
    logger.info(f"Train config: {vars(train_config)}")
    logger.info(f"Estimated parameters: {model_config.num_parameters():,}")

    train(model_config, train_config, resume_from=args.resume)


if __name__ == "__main__":
    main()
