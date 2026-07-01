"""
Reward models for RL fine-tuning of TinyStories.

Provides two reward strategies:

1. **Learned reward model**: A transformer backbone with a scalar value head,
   trained on human or GPT-4 preference data to predict story quality scores.

2. **Heuristic reward**: A zero-dependency rule-based scorer that measures
   vocabulary diversity, repetition penalty, sentence structure, and length
   appropriateness. Useful for experimentation without labeled preference data.

Reference: Ziegler et al. (2019), "Fine-Tuning Language Models from Human Preferences"
"""

import math
import re
from collections import Counter
from typing import Callable, Optional

import torch
import torch.nn as nn

from tinystories.config import ModelConfig
from tinystories.model.transformer import TinyStoriesModel


class RewardModel(nn.Module):
    """Learned reward model built on top of the TinyStories transformer.

    Replaces the language modeling head with a scalar reward head that
    maps the final hidden state to a single quality score. The backbone
    can be initialized from a pre-trained checkpoint.

    Architecture:
        Input tokens → TinyStories backbone → final hidden states
        → mean pooling over non-padding positions → Linear(d_model, 1) → scalar reward

    Args:
        config: Model configuration for the transformer backbone.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Transformer backbone (shared architecture with the policy)
        self.backbone = TinyStoriesModel(config)

        # Remove the LM head — we don't need logits
        # We keep the backbone's forward for hidden states only
        self.reward_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

        # Initialize reward head
        for module in self.reward_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract final hidden states from the backbone.

        Runs the transformer backbone up to (and including) the final
        layer norm, but skips the LM head projection.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).

        Returns:
            Hidden states of shape (batch_size, seq_len, d_model).
        """
        B, T = input_ids.size()
        device = input_ids.device

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.backbone.token_emb(input_ids)
        pos_emb = self.backbone.pos_emb(pos)
        x = self.backbone.emb_dropout(tok_emb + pos_emb)

        for block in self.backbone.blocks:
            x = block(x)

        x = self.backbone.ln_f(x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reward scores for input sequences.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Optional binary mask of shape (batch_size, seq_len).
                           1 for real tokens, 0 for padding. If None, all
                           positions are treated as real tokens.

        Returns:
            Reward scores of shape (batch_size,).
        """
        hidden = self._get_hidden_states(input_ids)  # (B, T, d_model)

        # Pool over sequence: mean of non-padding positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)  # (B, d_model)

        reward = self.reward_head(pooled).squeeze(-1)  # (B,)
        return reward

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: torch.device = None,
    ) -> "RewardModel":
        """Create a reward model from a pre-trained TinyStories checkpoint.

        Loads the transformer backbone weights from a standard training
        checkpoint and initializes the reward head from scratch.

        Args:
            checkpoint_path: Path to a TinyStories .pt checkpoint.
            device: Device to load the model onto.

        Returns:
            RewardModel with pre-trained backbone.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        config = ModelConfig(**checkpoint["model_config"])
        model = cls(config)

        # Load backbone weights (the backbone is a TinyStoriesModel)
        model.backbone.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        return model


# ============================================================================
# Heuristic (rule-based) reward function
# ============================================================================


def compute_heuristic_reward(
    text: str,
    prompt: str = "",
) -> float:
    """Compute a rule-based reward score for a generated story.

    Measures multiple aspects of text quality without requiring a trained
    reward model. The score combines:

    1. **Vocabulary diversity**: Type-token ratio (unique words / total words)
    2. **Repetition penalty**: Penalizes repeated n-grams (bigrams, trigrams)
    3. **Sentence structure**: Rewards proper sentence endings and variety
    4. **Length appropriateness**: Penalizes too-short or too-long outputs
    5. **Coherence heuristics**: Penalizes all-caps, excessive punctuation

    All sub-scores are normalized to [0, 1] and combined as a weighted sum,
    then scaled to roughly [-1, 1] for RL training.

    Args:
        text: The full generated text (including the prompt).
        prompt: The original prompt (subtracted to score only the generation).

    Returns:
        Scalar reward in approximately [-1, 1].
    """
    # Strip prompt to evaluate only the generated continuation
    generated = text[len(prompt):].strip() if prompt else text.strip()

    if len(generated) < 5:
        return -1.0

    words = generated.lower().split()
    num_words = len(words)

    if num_words < 3:
        return -0.8

    # --- Vocabulary diversity (type-token ratio) ---
    unique_words = len(set(words))
    ttr = unique_words / num_words
    diversity_score = min(ttr / 0.7, 1.0)  # 0.7 TTR ≈ excellent diversity

    # --- Repetition penalty ---
    # Penalize repeated bigrams and trigrams
    def ngram_repetition_rate(tokens, n):
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        return repeated / max(len(ngrams), 1)

    bigram_rep = ngram_repetition_rate(words, 2)
    trigram_rep = ngram_repetition_rate(words, 3)
    repetition_score = max(0.0, 1.0 - 2.0 * bigram_rep - 3.0 * trigram_rep)

    # --- Sentence structure ---
    sentences = re.split(r"[.!?]+", generated)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = len(sentences)

    # Reward having multiple sentences
    sentence_count_score = min(num_sentences / 4.0, 1.0)

    # Reward sentence length variety
    if num_sentences > 1:
        sent_lengths = [len(s.split()) for s in sentences]
        mean_len = sum(sent_lengths) / len(sent_lengths)
        variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
        length_variety_score = min(math.sqrt(variance) / 5.0, 1.0)
    else:
        length_variety_score = 0.0

    # --- Length appropriateness ---
    # Sweet spot: 30-150 words for a short children's story
    if num_words < 10:
        length_score = 0.2
    elif num_words < 30:
        length_score = 0.5 + 0.5 * (num_words - 10) / 20
    elif num_words <= 150:
        length_score = 1.0
    elif num_words <= 250:
        length_score = 1.0 - 0.5 * (num_words - 150) / 100
    else:
        length_score = 0.3

    # --- Coherence heuristics ---
    # Penalize excessive capitalization
    upper_ratio = sum(1 for c in generated if c.isupper()) / max(
        len(generated), 1
    )
    caps_penalty = max(0.0, 1.0 - 5.0 * max(0, upper_ratio - 0.15))

    # Penalize excessive special characters / repetitive punctuation
    special_ratio = sum(
        1 for c in generated if c in "!?*#@^&()[]{}|~"
    ) / max(len(generated), 1)
    special_penalty = max(0.0, 1.0 - 10.0 * special_ratio)

    # --- Combine scores ---
    reward = (
        0.25 * diversity_score
        + 0.25 * repetition_score
        + 0.10 * sentence_count_score
        + 0.10 * length_variety_score
        + 0.15 * length_score
        + 0.075 * caps_penalty
        + 0.075 * special_penalty
    )

    # Scale from [0, 1] to [-1, 1]
    reward = 2.0 * reward - 1.0

    return reward


def make_reward_fn(
    reward_model: Optional[RewardModel] = None,
    tokenizer=None,
    use_heuristic: bool = True,
    heuristic_weight: float = 0.3,
    device: torch.device = None,
) -> Callable:
    """Create a reward function for PPO training.

    If a learned reward model is provided, uses it as the primary signal.
    Optionally blends with the heuristic reward for regularization.

    Args:
        reward_model: Optional learned reward model.
        tokenizer: Tokenizer for decoding (needed for heuristic reward).
        use_heuristic: Whether to use the heuristic reward.
        heuristic_weight: Weight of heuristic when blending with learned reward.
        device: Device for the reward model.

    Returns:
        Callable that takes (input_ids, prompts) and returns reward tensor.
    """

    def reward_fn(
        input_ids: torch.Tensor,
        prompts: list,
    ) -> torch.Tensor:
        """Compute rewards for a batch of generated sequences.

        Args:
            input_ids: Generated token IDs of shape (batch_size, seq_len).
            prompts: List of original prompt strings.

        Returns:
            Reward tensor of shape (batch_size,).
        """
        batch_size = input_ids.size(0)
        rewards = torch.zeros(batch_size, device=input_ids.device)

        # Learned reward
        if reward_model is not None:
            with torch.no_grad():
                learned_reward = reward_model(input_ids)
            if use_heuristic and tokenizer is not None:
                # Blend learned and heuristic
                heuristic_rewards = []
                for i in range(batch_size):
                    text = tokenizer.decode(input_ids[i].tolist())
                    h_reward = compute_heuristic_reward(text, prompts[i])
                    heuristic_rewards.append(h_reward)
                h_tensor = torch.tensor(
                    heuristic_rewards, device=input_ids.device, dtype=torch.float
                )
                rewards = (
                    (1 - heuristic_weight) * learned_reward
                    + heuristic_weight * h_tensor
                )
            else:
                rewards = learned_reward

        # Pure heuristic reward
        elif use_heuristic and tokenizer is not None:
            heuristic_rewards = []
            for i in range(batch_size):
                text = tokenizer.decode(input_ids[i].tolist())
                h_reward = compute_heuristic_reward(text, prompts[i])
                heuristic_rewards.append(h_reward)
            rewards = torch.tensor(
                heuristic_rewards, device=input_ids.device, dtype=torch.float
            )

        return rewards

    return reward_fn
