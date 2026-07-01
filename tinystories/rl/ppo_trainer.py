"""
Proximal Policy Optimization (PPO) trainer for TinyStories.

Implements the core PPO training algorithm that alternates between:
1. **Rollout phase**: Generate text from the policy and collect rewards
2. **Optimization phase**: Update the policy using clipped surrogate objective

The trainer supports:
- Clipped policy loss (standard PPO-Clip)
- Value function loss with optional clipping
- Entropy bonus for exploration
- KL penalty against a frozen reference model (adaptive coefficient)
- Gradient accumulation and mixed precision training
- Periodic checkpointing and W&B logging

Reference:
- Schulman et al. (2017), "Proximal Policy Optimization Algorithms"
- Ziegler et al. (2019), "Fine-Tuning Language Models from Human Preferences"
- Ouyang et al. (2022), "Training language models to follow instructions with human feedback"
"""

import copy
import logging
import math
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from tinystories.config import ModelConfig, RLConfig
from tinystories.rl.rollout import (
    RolloutBuffer,
    compute_logprobs_from_logits,
    generate_rollouts,
)
from tinystories.rl.value import PolicyWithValueHead

logger = logging.getLogger(__name__)


class PPOTrainer:
    """Proximal Policy Optimization trainer for language models.

    Manages the full RL fine-tuning loop: rollout generation, advantage
    estimation, and multi-epoch mini-batch PPO updates.

    Args:
        policy_model: PolicyWithValueHead being optimized.
        ref_model: Frozen reference model for KL computation.
        reward_fn: Callable mapping (input_ids, prompts) → reward tensor.
        tokenizer: Tokenizer for encoding/decoding text.
        rl_config: PPO and training hyperparameters.
        prompts: List of training prompts for rollout generation.
        device: Device for computation.
    """

    def __init__(
        self,
        policy_model: PolicyWithValueHead,
        ref_model,
        reward_fn: Callable,
        tokenizer,
        rl_config: RLConfig,
        prompts: List[str],
        device: torch.device,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.rl_config = rl_config
        self.prompts = prompts
        self.device = device

        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer: AdamW with weight decay on 2D+ params only
        decay_params = []
        no_decay_params = []
        for name, param in self.policy_model.named_parameters():
            if param.requires_grad:
                if param.ndim >= 2:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": rl_config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=rl_config.learning_rate,
        )

        # Mixed precision
        self.use_amp = rl_config.use_amp and device.type == "cuda"
        self.amp_dtype = (
            torch.float16 if rl_config.dtype == "float16" else torch.bfloat16
        )
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.use_amp)
            if self.use_amp
            else None
        )

        # Adaptive KL coefficient
        self.kl_coeff = rl_config.kl_coeff

        # Tracking
        self.global_step = 0
        self.best_mean_reward = float("-inf")

    def _get_lr(self, step: int, total_steps: int) -> float:
        """Compute learning rate with warmup and cosine decay."""
        peak_lr = self.rl_config.learning_rate
        min_lr = peak_lr * 0.1

        if step < self.rl_config.warmup_steps:
            return peak_lr * (step + 1) / self.rl_config.warmup_steps

        decay_steps = total_steps - self.rl_config.warmup_steps
        progress = (step - self.rl_config.warmup_steps) / max(1, decay_steps)
        progress = min(progress, 1.0)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def compute_ppo_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        logits: torch.Tensor,
    ) -> dict:
        """Compute the PPO objective: policy loss + value loss + entropy bonus.

        The clipped surrogate objective prevents destructively large policy
        updates by clipping the probability ratio:
            L_clip = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)

        Args:
            logprobs: Current policy log-probs, shape (batch, gen_len).
            old_logprobs: Log-probs from the rollout phase, shape (batch, gen_len).
            advantages: GAE advantages, shape (batch, gen_len).
            values: Current value estimates, shape (batch, gen_len).
            old_values: Values from the rollout phase, shape (batch, gen_len).
            returns: Target returns, shape (batch, gen_len).
            logits: Current logits for entropy, shape (batch, gen_len, vocab).

        Returns:
            Dict with 'total_loss', 'policy_loss', 'value_loss', 'entropy',
            'approx_kl', and 'clip_fraction'.
        """
        clip_range = self.rl_config.clip_range

        # --- Policy (actor) loss ---
        ratio = torch.exp(logprobs - old_logprobs)  # r_t = π_θ / π_θ_old
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

        policy_loss_unclipped = -advantages * ratio
        policy_loss_clipped = -advantages * clipped_ratio
        policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

        # --- Value (critic) loss ---
        if self.rl_config.clip_value:
            # Clipped value loss
            value_clipped = old_values + torch.clamp(
                values - old_values, -clip_range, clip_range
            )
            value_loss_unclipped = (values - returns) ** 2
            value_loss_clipped = (value_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(
                value_loss_unclipped, value_loss_clipped
            ).mean()
        else:
            value_loss = 0.5 * F.mse_loss(values, returns)

        # --- Entropy bonus ---
        # Encourages exploration by rewarding higher-entropy action distributions
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # --- KL divergence (approximate) ---
        # Used for monitoring and adaptive KL penalty
        approx_kl = (old_logprobs - logprobs).mean()

        # --- Clip fraction (monitoring metric) ---
        clip_fraction = (
            (torch.abs(ratio - 1.0) > clip_range).float().mean().item()
        )

        # --- Total loss ---
        total_loss = (
            policy_loss
            + self.rl_config.value_loss_coeff * value_loss
            - self.rl_config.entropy_coeff * entropy
            + self.kl_coeff * approx_kl
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def _update_kl_coeff(self, mean_kl: float):
        """Adaptively adjust the KL penalty coefficient.

        If KL divergence exceeds the target, increase the penalty.
        If it's below, decrease it. This keeps the policy from
        drifting too far from the reference while still allowing learning.
        """
        if self.rl_config.kl_target is None:
            return

        if mean_kl > self.rl_config.kl_target * 1.5:
            self.kl_coeff *= 1.5
        elif mean_kl < self.rl_config.kl_target / 1.5:
            self.kl_coeff *= 1 / 1.5

        # Clamp to reasonable range
        self.kl_coeff = max(0.001, min(self.kl_coeff, 10.0))

    def train_step(self, rollout: RolloutBuffer) -> dict:
        """Perform PPO optimization on a single rollout buffer.

        Runs multiple epochs of mini-batch gradient updates using the
        collected rollout data.

        Args:
            rollout: Populated RolloutBuffer from generate_rollouts().

        Returns:
            Dict of averaged training metrics.
        """
        batch_size = len(rollout)
        if batch_size == 0:
            return {"error": "empty_rollout"}

        gen_len = rollout.logprobs.size(1)
        metrics_accum = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "total_loss": 0.0,
        }
        num_updates = 0

        for epoch in range(self.rl_config.ppo_epochs):
            # Shuffle indices for mini-batching
            indices = torch.randperm(batch_size, device=self.device)
            mini_batch_size = min(self.rl_config.mini_batch_size, batch_size)

            for start in range(0, batch_size, mini_batch_size):
                end = min(start + mini_batch_size, batch_size)
                mb_indices = indices[start:end]

                # Gather mini-batch data
                mb_full_ids = rollout.full_ids[mb_indices]
                mb_response_ids = rollout.response_ids[mb_indices]
                mb_old_logprobs = rollout.logprobs[mb_indices]
                mb_old_values = rollout.values[mb_indices]
                mb_advantages = rollout.advantages[mb_indices]
                mb_returns = rollout.returns[mb_indices]

                prompt_len = rollout.query_ids.size(1)

                # Forward pass under the current policy
                if self.use_amp:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                        logits, values, _ = self.policy_model(mb_full_ids)
                else:
                    logits, values, _ = self.policy_model(mb_full_ids)

                # Extract response portion
                response_logits = logits[:, prompt_len - 1:-1, :]
                response_values = values[:, prompt_len - 1:-1]

                actual_gen_len = min(
                    response_logits.size(1), mb_response_ids.size(1)
                )
                response_logits = response_logits[:, :actual_gen_len, :]
                response_values = response_values[:, :actual_gen_len]
                mb_response_ids_trimmed = mb_response_ids[:, :actual_gen_len]

                # Current log-probs
                current_logprobs = compute_logprobs_from_logits(
                    response_logits, mb_response_ids_trimmed
                )

                # Trim stored tensors to match
                mb_old_logprobs_t = mb_old_logprobs[:, :actual_gen_len]
                mb_old_values_t = mb_old_values[:, :actual_gen_len]
                mb_advantages_t = mb_advantages[:, :actual_gen_len]
                mb_returns_t = mb_returns[:, :actual_gen_len]

                # Compute PPO loss
                loss_dict = self.compute_ppo_loss(
                    logprobs=current_logprobs,
                    old_logprobs=mb_old_logprobs_t,
                    advantages=mb_advantages_t,
                    values=response_values,
                    old_values=mb_old_values_t,
                    returns=mb_returns_t,
                    logits=response_logits,
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_loss = loss_dict["total_loss"]

                if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(),
                        self.rl_config.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(),
                        self.rl_config.max_grad_norm,
                    )
                    self.optimizer.step()

                # Accumulate metrics
                for key in metrics_accum:
                    val = loss_dict[key]
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    metrics_accum[key] += val
                num_updates += 1

        # Average metrics
        if num_updates > 0:
            for key in metrics_accum:
                metrics_accum[key] /= num_updates

        # Adaptive KL coefficient
        self._update_kl_coeff(metrics_accum["approx_kl"])
        metrics_accum["kl_coeff"] = self.kl_coeff

        return metrics_accum

    def save_checkpoint(
        self,
        step: int,
        mean_reward: float,
        is_best: bool = False,
    ):
        """Save an RL training checkpoint.

        Args:
            step: Current rollout step.
            mean_reward: Mean reward of the latest rollout.
            is_best: Whether this is the best checkpoint so far.
        """
        out_path = Path(self.rl_config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "policy_state_dict": self.policy_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": vars(self.policy_model.config),
            "rl_config": vars(self.rl_config),
            "step": step,
            "mean_reward": mean_reward,
            "kl_coeff": self.kl_coeff,
        }

        ckpt_path = out_path / f"rl_checkpoint_step{step}.pt"
        torch.save(checkpoint, str(ckpt_path))
        logger.info(f"Saved RL checkpoint to {ckpt_path}")

        if is_best:
            best_path = out_path / "rl_best.pt"
            torch.save(checkpoint, str(best_path))
            logger.info(f"New best RL model! mean_reward={mean_reward:.4f}")

        latest_path = out_path / "rl_latest.pt"
        torch.save(checkpoint, str(latest_path))

    def train(self):
        """Main PPO training loop.

        Alternates between rollout generation and PPO optimization:
        1. Sample a batch of prompts
        2. Generate responses and compute rewards
        3. Run multiple PPO epochs on the collected data
        4. Log metrics and save checkpoints
        """
        total_steps = self.rl_config.total_rollout_steps
        logger.info(f"Starting PPO training for {total_steps} rollout steps")

        # Optional W&B logging
        wandb_run = None
        if self.rl_config.wandb_project:
            try:
                import wandb
                wandb_run = wandb.init(
                    project=self.rl_config.wandb_project,
                    name=self.rl_config.wandb_run_name or "ppo-tinystories",
                    config=vars(self.rl_config),
                )
            except ImportError:
                logger.warning("wandb not installed. Skipping W&B logging.")

        for step in range(total_steps):
            t0 = time.time()

            # Update learning rate
            lr = self._get_lr(step, total_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Sample prompts for this rollout
            batch_size = min(
                self.rl_config.rollout_batch_size, len(self.prompts)
            )
            prompt_indices = np.random.choice(
                len(self.prompts), size=batch_size, replace=False
            )
            batch_prompts = [self.prompts[i] for i in prompt_indices]

            # Generate rollouts
            rollout = generate_rollouts(
                policy_model=self.policy_model,
                ref_model=self.ref_model,
                reward_fn=self.reward_fn,
                tokenizer=self.tokenizer,
                prompts=batch_prompts,
                rl_config=self.rl_config,
                device=self.device,
            )

            if len(rollout) == 0:
                logger.warning(f"Step {step}: empty rollout, skipping.")
                continue

            mean_reward = rollout.rewards.mean().item()
            std_reward = rollout.rewards.std().item()

            # Compute KL divergence from reference
            kl_div = (rollout.logprobs - rollout.ref_logprobs).mean().item()

            # PPO optimization
            metrics = self.train_step(rollout)

            dt = time.time() - t0
            self.global_step = step

            # Logging
            if step % self.rl_config.log_every_steps == 0:
                logger.info(
                    f"step {step}/{total_steps} | "
                    f"reward {mean_reward:.3f}±{std_reward:.3f} | "
                    f"kl {kl_div:.4f} | "
                    f"policy_loss {metrics.get('policy_loss', 0):.4f} | "
                    f"value_loss {metrics.get('value_loss', 0):.4f} | "
                    f"entropy {metrics.get('entropy', 0):.4f} | "
                    f"kl_coeff {self.kl_coeff:.4f} | "
                    f"lr {lr:.2e} | "
                    f"{dt:.1f}s/step"
                )

                if wandb_run:
                    wandb_run.log(
                        {
                            "rl/mean_reward": mean_reward,
                            "rl/std_reward": std_reward,
                            "rl/kl_divergence": kl_div,
                            "rl/policy_loss": metrics.get("policy_loss", 0),
                            "rl/value_loss": metrics.get("value_loss", 0),
                            "rl/entropy": metrics.get("entropy", 0),
                            "rl/kl_coeff": self.kl_coeff,
                            "rl/clip_fraction": metrics.get("clip_fraction", 0),
                            "rl/lr": lr,
                        },
                        step=step,
                    )

            # Checkpointing
            is_best = mean_reward > self.best_mean_reward
            if is_best:
                self.best_mean_reward = mean_reward

            if step % self.rl_config.save_every_steps == 0 and step > 0:
                self.save_checkpoint(step, mean_reward, is_best=is_best)

        # Final checkpoint
        self.save_checkpoint(total_steps, mean_reward, is_best=True)

        if wandb_run:
            wandb_run.finish()

        logger.info(
            f"PPO training complete! "
            f"Best mean reward: {self.best_mean_reward:.4f}"
        )
