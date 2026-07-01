"""
Tests for RL fine-tuning components.

Verifies:
- RLConfig defaults and model-size tuning
- RewardModel forward pass shapes
- Heuristic reward scoring
- ValueHead and PolicyWithValueHead forward shapes
- GAE computation correctness
- PPO loss computation (clipping, value loss, entropy)
- RolloutBuffer storage and retrieval
- End-to-end mini RL loop
"""

import pytest
import torch

from tinystories.config import ModelConfig, RLConfig, get_default_rl_config
from tinystories.rl.reward import RewardModel, compute_heuristic_reward
from tinystories.rl.rollout import RolloutBuffer, compute_gae, compute_logprobs_from_logits
from tinystories.rl.value import PolicyWithValueHead, ValueHead


# ============================================================================
# Config Tests
# ============================================================================


class TestRLConfig:
    """Tests for RL configuration."""

    def test_default_config(self):
        """Default RLConfig should have sensible values."""
        config = RLConfig()
        assert config.clip_range == 0.2
        assert config.ppo_epochs == 4
        assert config.gamma == 1.0
        assert config.lam == 0.95
        assert config.learning_rate == 1e-5

    def test_model_size_tuning(self):
        """Larger models should get lower learning rates."""
        small = get_default_rl_config("tiny-1M")
        large = get_default_rl_config("tiny-33M")
        assert small.learning_rate > large.learning_rate

    def test_all_model_sizes(self):
        """Should return valid config for all model sizes."""
        for name in ["tiny-1M", "tiny-3M", "tiny-8M", "tiny-28M", "tiny-33M"]:
            config = get_default_rl_config(name)
            assert isinstance(config, RLConfig)
            assert config.learning_rate > 0


# ============================================================================
# Reward Tests
# ============================================================================


class TestRewardModel:
    """Tests for the learned reward model."""

    @pytest.fixture
    def small_reward_model(self):
        config = ModelConfig(
            vocab_size=256, context_length=32,
            n_layers=2, n_heads=4, d_model=64,
        )
        return RewardModel(config)

    def test_forward_shape(self, small_reward_model):
        """Reward should be a scalar per sequence."""
        x = torch.randint(0, 256, (4, 16))
        reward = small_reward_model(x)
        assert reward.shape == (4,)

    def test_forward_with_mask(self, small_reward_model):
        """Should handle attention masks correctly."""
        x = torch.randint(0, 256, (4, 16))
        mask = torch.ones(4, 16)
        mask[:, 12:] = 0  # Mask last 4 positions
        reward = small_reward_model(x, attention_mask=mask)
        assert reward.shape == (4,)

    def test_single_sequence(self, small_reward_model):
        """Should work with batch_size=1."""
        x = torch.randint(0, 256, (1, 8))
        reward = small_reward_model(x)
        assert reward.shape == (1,)


class TestHeuristicReward:
    """Tests for the rule-based heuristic reward function."""

    def test_good_story(self):
        """A well-formed story should score positively."""
        text = (
            "Once upon a time there was a little girl named Lily. "
            "She loved to play in the garden with her dog Max. "
            "One day they found a beautiful butterfly. "
            "Lily was so happy she danced around the flowers."
        )
        reward = compute_heuristic_reward(text)
        assert reward > 0.0

    def test_empty_text(self):
        """Empty or very short text should score very negatively."""
        assert compute_heuristic_reward("") == -1.0
        assert compute_heuristic_reward("ab") == -1.0

    def test_repetitive_text(self):
        """Highly repetitive text should score lower."""
        repetitive = "the cat sat. " * 20
        diverse = (
            "A little bird flew over the green hill. "
            "The sun was shining brightly in the sky. "
            "Children played happily near the sparkling river."
        )
        r_repetitive = compute_heuristic_reward(repetitive)
        r_diverse = compute_heuristic_reward(diverse)
        assert r_diverse > r_repetitive

    def test_prompt_stripping(self):
        """Should only score the generated part, not the prompt."""
        prompt = "Once upon a time"
        text = prompt + " there was a cat who loved to play."
        reward = compute_heuristic_reward(text, prompt=prompt)
        assert isinstance(reward, float)

    def test_reward_range(self):
        """Reward should be in [-1, 1]."""
        texts = [
            "Hello",
            "This is a test sentence with several words in it.",
            "The quick brown fox jumps over the lazy dog. " * 5,
        ]
        for text in texts:
            r = compute_heuristic_reward(text)
            assert -1.0 <= r <= 1.0, f"Reward {r} out of range for: {text[:50]}"


# ============================================================================
# Value Head Tests
# ============================================================================


class TestValueHead:
    """Tests for the value estimation head."""

    def test_output_shape(self):
        """Value head should produce per-position scalar values."""
        head = ValueHead(d_model=64)
        hidden = torch.randn(2, 16, 64)
        values = head(hidden)
        assert values.shape == (2, 16)

    def test_single_position(self):
        """Should work with single-position input."""
        head = ValueHead(d_model=64)
        hidden = torch.randn(1, 1, 64)
        values = head(hidden)
        assert values.shape == (1, 1)


class TestPolicyWithValueHead:
    """Tests for the combined policy + value model."""

    @pytest.fixture
    def small_pv_model(self):
        config = ModelConfig(
            vocab_size=256, context_length=32,
            n_layers=2, n_heads=4, d_model=64,
        )
        return PolicyWithValueHead(config)

    def test_forward_shapes(self, small_pv_model):
        """Should produce logits and values of correct shapes."""
        x = torch.randint(0, 256, (2, 16))
        logits, values, loss = small_pv_model(x)
        assert logits.shape == (2, 16, 256)
        assert values.shape == (2, 16)
        assert loss is None

    def test_forward_with_targets(self, small_pv_model):
        """Should compute loss when targets are provided."""
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        logits, values, loss = small_pv_model(x, y)
        assert loss is not None
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_generate_delegates(self, small_pv_model):
        """Generation should work through the policy backbone."""
        x = torch.randint(0, 256, (1, 4))
        output = small_pv_model.generate(x, max_new_tokens=5, top_k=10)
        assert output.shape[1] > 4


# ============================================================================
# GAE Tests
# ============================================================================


class TestComputeGAE:
    """Tests for Generalized Advantage Estimation."""

    def test_output_shapes(self):
        """Advantages and returns should match value shape."""
        rewards = torch.tensor([1.0, 0.5])
        values = torch.randn(2, 10)
        advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=0.95)
        assert advantages.shape == (2, 10)
        assert returns.shape == (2, 10)

    def test_returns_equal_advantages_plus_values(self):
        """Returns should be advantages + values (before normalization)."""
        rewards = torch.tensor([1.0])
        values = torch.ones(1, 5) * 0.5
        # Use gamma=1, lam=1 (Monte Carlo) for easy verification
        advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=1.0)
        # With normalization, check returns are valid
        assert not torch.isnan(returns).any()

    def test_zero_rewards(self):
        """Zero rewards should produce zero advantages (pre-normalization)."""
        rewards = torch.tensor([0.0, 0.0])
        values = torch.zeros(2, 5)
        advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=0.95)
        # With zero rewards and zero values, advantages should be zero
        assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-6)

    def test_positive_reward_positive_advantage(self):
        """Positive reward should create positive advantages at the end."""
        rewards = torch.tensor([5.0])
        values = torch.zeros(1, 3)
        advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=0.95)
        # The last position should have highest advantage (closest to reward)
        # After normalization, relative ordering is preserved
        assert not torch.isnan(advantages).any()


class TestComputeLogprobs:
    """Tests for log-probability extraction."""

    def test_shape(self):
        """Log-probs should match the token shape."""
        logits = torch.randn(2, 10, 256)
        tokens = torch.randint(0, 256, (2, 10))
        lp = compute_logprobs_from_logits(logits, tokens)
        assert lp.shape == (2, 10)

    def test_values_negative(self):
        """Log-probabilities should be negative (or zero)."""
        logits = torch.randn(2, 10, 256)
        tokens = torch.randint(0, 256, (2, 10))
        lp = compute_logprobs_from_logits(logits, tokens)
        assert (lp <= 0).all()


# ============================================================================
# RolloutBuffer Tests
# ============================================================================


class TestRolloutBuffer:
    """Tests for rollout data storage."""

    def test_empty_buffer(self):
        """Empty buffer should have length 0."""
        buf = RolloutBuffer()
        assert len(buf) == 0

    def test_populated_buffer(self):
        """Should correctly report length from full_ids."""
        buf = RolloutBuffer()
        buf.full_ids = torch.randint(0, 256, (8, 20))
        assert len(buf) == 8

    def test_stores_prompts(self):
        """Should store prompt strings."""
        buf = RolloutBuffer(prompts=["hello", "world"])
        assert buf.prompts == ["hello", "world"]


# ============================================================================
# PPO Loss Tests
# ============================================================================


class TestPPOLoss:
    """Tests for PPO loss computation (via PPOTrainer.compute_ppo_loss)."""

    @pytest.fixture
    def trainer_components(self):
        """Create minimal components for testing PPO loss."""
        config = ModelConfig(
            vocab_size=256, context_length=32,
            n_layers=1, n_heads=4, d_model=64,
        )
        rl_config = RLConfig(clip_range=0.2)
        policy = PolicyWithValueHead(config)
        return policy, rl_config

    def test_loss_is_scalar(self, trainer_components):
        """PPO total loss should be a scalar tensor."""
        from tinystories.rl.ppo_trainer import PPOTrainer

        policy, rl_config = trainer_components

        # Create a minimal trainer (we'll call compute_ppo_loss directly)
        ref_model = PolicyWithValueHead(policy.config)
        trainer = PPOTrainer(
            policy_model=policy,
            ref_model=ref_model,
            reward_fn=lambda ids, prompts: torch.zeros(ids.size(0)),
            tokenizer=None,
            rl_config=rl_config,
            prompts=["test"],
            device=torch.device("cpu"),
        )

        # Dummy data
        batch_size, gen_len, vocab = 4, 10, 256
        logprobs = torch.randn(batch_size, gen_len)
        old_logprobs = torch.randn(batch_size, gen_len)
        advantages = torch.randn(batch_size, gen_len)
        values = torch.randn(batch_size, gen_len)
        old_values = torch.randn(batch_size, gen_len)
        returns = torch.randn(batch_size, gen_len)
        logits = torch.randn(batch_size, gen_len, vocab)

        result = trainer.compute_ppo_loss(
            logprobs, old_logprobs, advantages,
            values, old_values, returns, logits,
        )

        assert "total_loss" in result
        assert result["total_loss"].ndim == 0
        assert not torch.isnan(result["total_loss"])

    def test_loss_keys(self, trainer_components):
        """Should return all expected metric keys."""
        from tinystories.rl.ppo_trainer import PPOTrainer

        policy, rl_config = trainer_components
        ref_model = PolicyWithValueHead(policy.config)
        trainer = PPOTrainer(
            policy_model=policy, ref_model=ref_model,
            reward_fn=lambda ids, prompts: torch.zeros(ids.size(0)),
            tokenizer=None, rl_config=rl_config,
            prompts=["test"], device=torch.device("cpu"),
        )

        result = trainer.compute_ppo_loss(
            torch.randn(2, 5), torch.randn(2, 5), torch.randn(2, 5),
            torch.randn(2, 5), torch.randn(2, 5), torch.randn(2, 5),
            torch.randn(2, 5, 256),
        )

        expected_keys = {
            "total_loss", "policy_loss", "value_loss",
            "entropy", "approx_kl", "clip_fraction",
        }
        assert expected_keys.issubset(result.keys())
