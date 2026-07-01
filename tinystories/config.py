"""
Model and training configurations for TinyStories.

Defines preset architectures from the paper ranging from ~1M to ~33M parameters,
following the GPT-Neo decoder-only transformer design. Each config specifies
the full set of hyperparameters needed for training and inference.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer model architecture configuration.

    Follows the GPT-Neo style decoder-only transformer used in the TinyStories paper.
    The paper explores models from 1M to 33M parameters, demonstrating that even
    very small models can produce coherent English when trained on the right data.
    """

    # Architecture
    vocab_size: int = 50257  # GPT-2/GPT-Neo tokenizer vocabulary size
    context_length: int = 512  # Maximum sequence length (tokens)
    n_layers: int = 8  # Number of transformer blocks
    n_heads: int = 16  # Number of attention heads
    d_model: int = 128  # Hidden dimension / embedding size
    d_ff: Optional[int] = None  # Feed-forward inner dimension (default: 4 * d_model)

    # Regularization
    dropout: float = 0.1
    attn_dropout: float = 0.1

    # Positional encoding
    max_position_embeddings: int = 512

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )

    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    def num_parameters(self, count_embeddings: bool = True) -> int:
        """Estimate total parameter count.

        Returns an approximate count based on the standard transformer architecture:
        - Token embeddings: vocab_size * d_model
        - Position embeddings: max_position_embeddings * d_model
        - Per-layer: attention (4 * d_model^2) + FFN (2 * d_model * d_ff) + layernorms
        - Final layernorm + LM head (tied with token embeddings)
        """
        # Embeddings
        emb = self.vocab_size * self.d_model + self.max_position_embeddings * self.d_model

        # Per transformer block
        attn = 4 * self.d_model * self.d_model  # Q, K, V, output projections
        attn_bias = 4 * self.d_model
        ffn = 2 * self.d_model * self.d_ff  # up + down projections
        ffn_bias = self.d_model + self.d_ff
        ln = 4 * self.d_model  # 2 layernorms per block (weight + bias each)

        per_layer = attn + attn_bias + ffn + ffn_bias + ln
        layers_total = self.n_layers * per_layer

        # Final layernorm
        final_ln = 2 * self.d_model

        total = layers_total + final_ln
        if count_embeddings:
            total += emb

        return total


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Defaults follow common practices from TinyStories reproductions:
    AdamW optimizer with cosine LR schedule, mixed precision training,
    and gradient accumulation to simulate larger effective batch sizes.
    """

    # Data
    data_dir: str = "data/tinystories"
    output_dir: str = "checkpoints"

    # Optimization
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    # Schedule
    warmup_steps: int = 200
    lr_scheduler: str = "cosine"  # "cosine" or "linear"
    min_lr_ratio: float = 0.1  # Minimum LR as fraction of peak LR

    # Batch
    batch_size: int = 64
    gradient_accumulation_steps: int = 4

    # Training duration
    max_epochs: int = 10
    max_steps: Optional[int] = None  # If set, overrides max_epochs

    # Checkpointing
    save_every_steps: int = 1000
    eval_every_steps: int = 500
    log_every_steps: int = 50

    # Mixed precision
    use_amp: bool = True
    dtype: str = "float16"  # "float16" or "bfloat16"

    # Reproducibility
    seed: int = 42

    # Logging
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"


@dataclass
class RLConfig:
    """Reinforcement learning fine-tuning configuration (PPO).

    Hyperparameters for Proximal Policy Optimization used to fine-tune
    a pre-trained TinyStories model against a reward signal (learned or
    heuristic). The defaults follow standard PPO practices from the
    RLHF literature (Ziegler et al., 2019; Ouyang et al., 2022).
    """

    # PPO clipping and objectives
    clip_range: float = 0.2  # PPO clipping parameter ε
    clip_value: bool = True  # Whether to also clip the value loss
    value_loss_coeff: float = 0.5  # Weight of value loss in total objective
    entropy_coeff: float = 0.01  # Entropy bonus to encourage exploration

    # KL penalty against reference policy
    kl_coeff: float = 0.1  # Initial KL penalty coefficient
    kl_target: Optional[float] = 6.0  # Target KL divergence for adaptive coeff
    kl_horizon: int = 10000  # Steps over which to adapt kl_coeff

    # GAE (Generalized Advantage Estimation)
    gamma: float = 1.0  # Discount factor (1.0 = no discounting for short stories)
    lam: float = 0.95  # GAE λ parameter

    # PPO training
    ppo_epochs: int = 4  # Number of PPO epochs per rollout batch
    mini_batch_size: int = 8  # Mini-batch size for PPO updates
    max_gen_len: int = 128  # Maximum generation length during rollouts
    rollout_batch_size: int = 32  # Number of prompts per rollout
    total_rollout_steps: int = 1000  # Total number of rollout-update cycles

    # Optimization
    learning_rate: float = 1e-5  # Lower LR than pre-training for fine-tuning
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 50

    # Reward
    reward_model_path: Optional[str] = None  # Path to learned reward model
    use_heuristic_reward: bool = True  # Use rule-based reward as fallback

    # Reference model
    ref_model_path: Optional[str] = None  # Defaults to the SFT checkpoint

    # Checkpointing
    output_dir: str = "checkpoints/rl"
    save_every_steps: int = 100
    log_every_steps: int = 10
    eval_every_steps: int = 50

    # Mixed precision
    use_amp: bool = True
    dtype: str = "float16"

    # Reproducibility
    seed: int = 42

    # Logging
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Device
    device: str = "auto"


def get_default_rl_config(model_name: str = "tiny-3M") -> RLConfig:
    """Get RL config with defaults tuned for a given model size.

    Smaller models tolerate higher learning rates and larger KL budgets,
    while larger models need more conservative fine-tuning.
    """
    config = RLConfig()

    if "1M" in model_name or "1L" in model_name:
        config.learning_rate = 5e-5
        config.rollout_batch_size = 64
        config.mini_batch_size = 16
        config.kl_coeff = 0.05
    elif "3M" in model_name:
        config.learning_rate = 2e-5
        config.rollout_batch_size = 32
        config.mini_batch_size = 8
    elif "8M" in model_name:
        config.learning_rate = 1e-5
        config.rollout_batch_size = 32
        config.mini_batch_size = 8
    elif "28M" in model_name or "33M" in model_name:
        config.learning_rate = 5e-6
        config.rollout_batch_size = 16
        config.mini_batch_size = 4
        config.warmup_steps = 100
        config.kl_coeff = 0.2

    return config


# ============================================================================
# Preset Configurations from the Paper
# ============================================================================

# Table 1 in the paper: model configurations varying depth and width
# to study the emergence of language capabilities at different scales.

TINY_1M = ModelConfig(
    n_layers=4,
    n_heads=8,
    d_model=64,
    context_length=512,
    dropout=0.1,
)

TINY_3M = ModelConfig(
    n_layers=8,
    n_heads=16,
    d_model=128,
    context_length=512,
    dropout=0.1,
)

TINY_8M = ModelConfig(
    n_layers=8,
    n_heads=8,
    d_model=256,
    context_length=512,
    dropout=0.1,
)

TINY_28M = ModelConfig(
    n_layers=16,
    n_heads=8,
    d_model=512,
    context_length=512,
    dropout=0.1,
)

TINY_33M = ModelConfig(
    n_layers=16,
    n_heads=12,
    d_model=576,
    context_length=512,
    dropout=0.1,
)

# Single-layer experiment from the paper (Section 4.2):
# "even models with only one transformer block can produce coherent stories"
TINY_1L = ModelConfig(
    n_layers=1,
    n_heads=8,
    d_model=512,
    context_length=512,
    dropout=0.1,
)

MODEL_CONFIGS = {
    "tiny-1M": TINY_1M,
    "tiny-3M": TINY_3M,
    "tiny-8M": TINY_8M,
    "tiny-28M": TINY_28M,
    "tiny-33M": TINY_33M,
    "tiny-1L": TINY_1L,
}


def get_model_config(name: str) -> ModelConfig:
    """Retrieve a preset model configuration by name.

    Args:
        name: One of 'tiny-1M', 'tiny-3M', 'tiny-8M', 'tiny-28M', 'tiny-33M', 'tiny-1L'

    Returns:
        ModelConfig for the requested preset.

    Raises:
        ValueError: If the name is not recognized.
    """
    if name not in MODEL_CONFIGS:
        available = ", ".join(sorted(MODEL_CONFIGS.keys()))
        raise ValueError(f"Unknown model config '{name}'. Available: {available}")
    return MODEL_CONFIGS[name]


def get_default_train_config(model_name: str = "tiny-3M") -> TrainConfig:
    """Get training config with defaults tuned for a given model size.

    Smaller models use higher learning rates and less gradient accumulation,
    while larger models need more conservative hyperparameters.
    """
    config = TrainConfig()

    if "1M" in model_name or "1L" in model_name:
        config.learning_rate = 1e-3
        config.batch_size = 128
        config.gradient_accumulation_steps = 2
    elif "3M" in model_name:
        config.learning_rate = 5e-4
        config.batch_size = 64
        config.gradient_accumulation_steps = 4
    elif "8M" in model_name:
        config.learning_rate = 3e-4
        config.batch_size = 64
        config.gradient_accumulation_steps = 4
    elif "28M" in model_name or "33M" in model_name:
        config.learning_rate = 1e-4
        config.batch_size = 32
        config.gradient_accumulation_steps = 8
        config.warmup_steps = 500

    return config
