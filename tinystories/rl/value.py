"""
Value head and policy-value wrapper for PPO training.

In PPO, the policy network (the language model) is augmented with a value
head that estimates the expected cumulative reward from each token position.
This module provides:

1. ``ValueHead``: A small MLP that maps hidden states to scalar values.
2. ``PolicyWithValueHead``: Wraps a ``TinyStoriesModel`` (policy) and a
   ``ValueHead`` so that a single forward pass produces both next-token
   logits and per-position value estimates.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from tinystories.config import ModelConfig
from tinystories.model.transformer import TinyStoriesModel

logger = logging.getLogger(__name__)


class ValueHead(nn.Module):
    """Scalar value head for estimating state values.

    Maps the transformer's hidden states to per-position scalar value
    estimates used in GAE and the PPO value loss.

    Architecture:
        hidden_states → Linear(d_model, d_model) → ReLU
                      → Linear(d_model, 1) → scalar value

    Args:
        d_model: Hidden dimension of the transformer backbone.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for stable early training."""
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute value estimates from hidden states.

        Args:
            hidden_states: Shape (batch_size, seq_len, d_model).

        Returns:
            Values of shape (batch_size, seq_len).
        """
        return self.head(hidden_states).squeeze(-1)


class PolicyWithValueHead(nn.Module):
    """Policy model augmented with a value head for PPO.

    Combines the language model (policy) and a value estimator into a
    single module. During the forward pass, hidden states are shared
    between the LM head (producing logits) and the value head (producing
    per-token value estimates), avoiding redundant computation.

    Args:
        config: Model configuration for the transformer backbone.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.policy = TinyStoriesModel(config)
        self.value_head = ValueHead(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass producing logits, values, and optionally loss.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            targets: Optional target IDs for computing cross-entropy loss.

        Returns:
            Tuple of (logits, values, loss):
            - logits: Shape (batch_size, seq_len, vocab_size)
            - values: Shape (batch_size, seq_len)
            - loss: Scalar CE loss if targets provided, else None
        """
        B, T = input_ids.size()
        device = input_ids.device

        # Shared forward: compute hidden states once
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.policy.token_emb(input_ids)
        pos_emb = self.policy.pos_emb(pos)
        x = self.policy.emb_dropout(tok_emb + pos_emb)

        for block in self.policy.blocks:
            x = block(x)

        hidden = self.policy.ln_f(x)  # (B, T, d_model)

        # Policy head: logits for next-token prediction
        logits = self.policy.lm_head(hidden)  # (B, T, vocab_size)

        # Value head: per-token value estimates
        values = self.value_head(hidden)  # (B, T)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            import torch.nn.functional as F
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, values, loss

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: torch.device = None,
    ) -> "PolicyWithValueHead":
        """Load policy from a pre-trained SFT checkpoint.

        The transformer backbone (policy) is initialized from the checkpoint.
        The value head is initialized from scratch.

        Args:
            checkpoint_path: Path to a TinyStories .pt checkpoint.
            device: Device to load the model onto.

        Returns:
            PolicyWithValueHead with pre-trained policy backbone.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        config = ModelConfig(**checkpoint["model_config"])
        model = cls(config)

        # Load policy weights
        model.policy.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        logger.info(
            f"Loaded PolicyWithValueHead from {checkpoint_path} "
            f"(step={checkpoint.get('step', '?')})"
        )

        return model

    def generate(self, *args, **kwargs):
        """Delegate generation to the underlying policy model."""
        return self.policy.generate(*args, **kwargs)
