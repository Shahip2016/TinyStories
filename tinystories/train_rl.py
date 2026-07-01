"""
CLI entry point for RL fine-tuning of TinyStories models.

Fine-tunes a pre-trained (SFT) TinyStories model using Proximal Policy
Optimization (PPO) against a reward signal. Supports both learned reward
models and a zero-dependency heuristic reward function.

Usage:
    # Fine-tune with heuristic reward (no extra data needed)
    python -m tinystories.train_rl \\
        --checkpoint checkpoints/tiny-3M/best.pt \\
        --use_heuristic_reward \\
        --output_dir checkpoints/rl-tiny-3M \\
        --total_steps 500

    # Fine-tune with a learned reward model
    python -m tinystories.train_rl \\
        --checkpoint checkpoints/tiny-3M/best.pt \\
        --reward_model checkpoints/reward_model.pt \\
        --output_dir checkpoints/rl-tiny-3M

Reference: PPO (Schulman et al., 2017), RLHF (Ouyang et al., 2022)
"""

import argparse
import copy
import logging

import numpy as np
import torch

from tinystories.config import ModelConfig, RLConfig, get_default_rl_config
from tinystories.generate import DEFAULT_PROMPTS
from tinystories.rl.ppo_trainer import PPOTrainer
from tinystories.rl.reward import RewardModel, make_reward_fn
from tinystories.rl.value import PolicyWithValueHead
from tinystories.train import get_device

logger = logging.getLogger(__name__)


def train_rl(
    checkpoint_path: str,
    rl_config: RLConfig,
    reward_model_path: str = None,
    prompts: list = None,
):
    """Run RL fine-tuning on a pre-trained TinyStories model.

    Args:
        checkpoint_path: Path to the SFT checkpoint to fine-tune.
        rl_config: RL training configuration.
        reward_model_path: Optional path to a learned reward model.
        prompts: List of training prompts. Defaults to paper prompts.
    """
    device = get_device(rl_config.device)
    logger.info(f"RL training on device: {device}")

    # Set random seeds
    torch.manual_seed(rl_config.seed)
    np.random.seed(rl_config.seed)

    # Load the policy model (with value head) from SFT checkpoint
    logger.info(f"Loading policy from {checkpoint_path}")
    policy_model = PolicyWithValueHead.from_pretrained(checkpoint_path, device)

    # Create frozen reference model (deep copy of the policy backbone)
    logger.info("Creating frozen reference model")
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    # Set up reward function
    reward_model = None
    if reward_model_path:
        logger.info(f"Loading reward model from {reward_model_path}")
        reward_model = RewardModel.from_pretrained(reward_model_path, device)
        reward_model.eval()

    reward_fn = make_reward_fn(
        reward_model=reward_model,
        tokenizer=tokenizer,
        use_heuristic=rl_config.use_heuristic_reward,
        device=device,
    )

    # Training prompts
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    logger.info(f"Training with {len(prompts)} prompts")
    logger.info(f"RL config: {vars(rl_config)}")

    # Create PPO trainer
    trainer = PPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_fn=reward_fn,
        tokenizer=tokenizer,
        rl_config=rl_config,
        prompts=prompts,
        device=device,
    )

    # Run training
    trainer.train()

    logger.info("RL fine-tuning complete!")
    return policy_model


def main():
    parser = argparse.ArgumentParser(
        description="RL fine-tune a TinyStories model using PPO"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to pre-trained SFT checkpoint (.pt file)",
    )
    parser.add_argument(
        "--reward_model", type=str, default=None,
        help="Path to learned reward model checkpoint",
    )
    parser.add_argument(
        "--use_heuristic_reward", action="store_true", default=True,
        help="Use heuristic reward function (default: True)",
    )
    parser.add_argument(
        "--no_heuristic_reward", action="store_true",
        help="Disable heuristic reward (use only learned reward)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints/rl",
        help="Directory to save RL checkpoints",
    )
    parser.add_argument(
        "--total_steps", type=int, default=None,
        help="Total number of rollout steps",
    )
    parser.add_argument(
        "--rollout_batch_size", type=int, default=None,
        help="Number of prompts per rollout batch",
    )
    parser.add_argument(
        "--ppo_epochs", type=int, default=None,
        help="PPO epochs per rollout",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate override",
    )
    parser.add_argument(
        "--clip_range", type=float, default=None,
        help="PPO clip range override",
    )
    parser.add_argument(
        "--kl_coeff", type=float, default=None,
        help="Initial KL penalty coefficient",
    )
    parser.add_argument(
        "--max_gen_len", type=int, default=None,
        help="Maximum generation length during rollouts",
    )
    parser.add_argument(
        "--prompt_file", type=str, default=None,
        help="File with training prompts (one per line)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto/cuda/cpu/mps",
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="W&B project name for logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Build RL config
    rl_config = get_default_rl_config()
    rl_config.output_dir = args.output_dir
    rl_config.seed = args.seed
    rl_config.device = args.device
    rl_config.use_heuristic_reward = args.use_heuristic_reward and not args.no_heuristic_reward

    if args.reward_model:
        rl_config.reward_model_path = args.reward_model
    if args.total_steps:
        rl_config.total_rollout_steps = args.total_steps
    if args.rollout_batch_size:
        rl_config.rollout_batch_size = args.rollout_batch_size
    if args.ppo_epochs:
        rl_config.ppo_epochs = args.ppo_epochs
    if args.lr:
        rl_config.learning_rate = args.lr
    if args.clip_range:
        rl_config.clip_range = args.clip_range
    if args.kl_coeff:
        rl_config.kl_coeff = args.kl_coeff
    if args.max_gen_len:
        rl_config.max_gen_len = args.max_gen_len
    if args.wandb_project:
        rl_config.wandb_project = args.wandb_project

    # Load prompts
    prompts = None
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

    train_rl(
        checkpoint_path=args.checkpoint,
        rl_config=rl_config,
        reward_model_path=args.reward_model,
        prompts=prompts,
    )


if __name__ == "__main__":
    main()
