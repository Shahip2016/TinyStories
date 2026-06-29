"""
Tests for TinyStories model architecture, config, and data pipeline.

Verifies:
- Model configurations produce correct parameter counts
- Transformer forward pass shapes are correct
- Causal attention mask works properly
- Tokenization and dataset loading work
- Generation produces valid outputs
"""

import pytest
import torch

from tinystories.config import ModelConfig, get_model_config, MODEL_CONFIGS
from tinystories.model.attention import CausalSelfAttention
from tinystories.model.transformer import (
    FeedForward,
    TinyStoriesModel,
    TransformerBlock,
)


# ============================================================================
# Config Tests
# ============================================================================


class TestModelConfig:
    """Tests for model configuration system."""

    def test_all_presets_exist(self):
        """All documented preset names should be retrievable."""
        for name in ["tiny-1M", "tiny-3M", "tiny-8M", "tiny-28M", "tiny-33M", "tiny-1L"]:
            config = get_model_config(name)
            assert isinstance(config, ModelConfig)

    def test_invalid_config_raises(self):
        """Requesting an unknown config should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model config"):
            get_model_config("nonexistent-config")

    def test_d_head_computation(self):
        """d_head should equal d_model // n_heads."""
        config = ModelConfig(d_model=256, n_heads=8)
        assert config.d_head == 32

    def test_d_ff_default(self):
        """d_ff should default to 4 * d_model."""
        config = ModelConfig(d_model=128)
        assert config.d_ff == 512

    def test_d_ff_custom(self):
        """Custom d_ff should override the default."""
        config = ModelConfig(d_model=128, d_ff=256)
        assert config.d_ff == 256

    def test_invalid_head_divisibility(self):
        """d_model not divisible by n_heads should raise AssertionError."""
        with pytest.raises(AssertionError):
            ModelConfig(d_model=100, n_heads=3)

    def test_parameter_count_positive(self):
        """Parameter count should always be positive."""
        for name, config in MODEL_CONFIGS.items():
            assert config.num_parameters() > 0, f"{name} has non-positive param count"

    def test_parameter_count_ordering(self):
        """Larger configs should have more parameters."""
        p1 = get_model_config("tiny-1M").num_parameters()
        p3 = get_model_config("tiny-3M").num_parameters()
        p8 = get_model_config("tiny-8M").num_parameters()
        assert p1 < p3 < p8


# ============================================================================
# Attention Tests
# ============================================================================


class TestCausalSelfAttention:
    """Tests for the multi-head causal self-attention module."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        attn = CausalSelfAttention(d_model=64, n_heads=4, max_seq_len=32)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_causal_masking(self):
        """Attention should only attend to current and past positions."""
        attn = CausalSelfAttention(d_model=64, n_heads=4, max_seq_len=32)
        # The causal mask should be lower-triangular
        mask = attn.causal_mask[0, 0, :8, :8]
        for i in range(8):
            for j in range(8):
                if j <= i:
                    assert mask[i, j] == 1.0, f"Position ({i},{j}) should be visible"
                else:
                    assert mask[i, j] == 0.0, f"Position ({i},{j}) should be masked"

    def test_single_token(self):
        """Should handle single-token sequences."""
        attn = CausalSelfAttention(d_model=64, n_heads=4, max_seq_len=32)
        x = torch.randn(1, 1, 64)
        out = attn(x)
        assert out.shape == (1, 1, 64)

    def test_different_head_counts(self):
        """Should work with various head configurations."""
        for n_heads in [1, 2, 4, 8]:
            attn = CausalSelfAttention(d_model=64, n_heads=n_heads, max_seq_len=16)
            x = torch.randn(1, 8, 64)
            out = attn(x)
            assert out.shape == (1, 8, 64)


# ============================================================================
# FeedForward Tests
# ============================================================================


class TestFeedForward:
    """Tests for the position-wise feed-forward network."""

    def test_output_shape(self):
        """Output dimension should match input dimension."""
        ffn = FeedForward(d_model=128, d_ff=512)
        x = torch.randn(2, 16, 128)
        out = ffn(x)
        assert out.shape == (2, 16, 128)

    def test_custom_d_ff(self):
        """Should work with non-standard inner dimensions."""
        ffn = FeedForward(d_model=64, d_ff=128)
        x = torch.randn(1, 4, 64)
        out = ffn(x)
        assert out.shape == (1, 4, 64)


# ============================================================================
# TransformerBlock Tests
# ============================================================================


class TestTransformerBlock:
    """Tests for the pre-norm transformer block."""

    def test_output_shape(self):
        """Block output should have same shape as input."""
        config = ModelConfig(d_model=64, n_heads=4, n_layers=1, context_length=32)
        block = TransformerBlock(config)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 16, 64)

    def test_residual_connection(self):
        """Output should differ from input (residual adds something)."""
        config = ModelConfig(d_model=64, n_heads=4, n_layers=1, context_length=32)
        block = TransformerBlock(config)
        block.eval()
        x = torch.randn(2, 16, 64)
        out = block(x)
        # The residual means output ≠ input (unless weights are degenerate)
        assert not torch.allclose(x, out, atol=1e-6)


# ============================================================================
# Full Model Tests
# ============================================================================


class TestTinyStoriesModel:
    """Tests for the full TinyStories language model."""

    @pytest.fixture
    def small_model(self):
        """Create a tiny model for testing."""
        config = ModelConfig(
            vocab_size=256,
            context_length=32,
            n_layers=2,
            n_heads=4,
            d_model=64,
        )
        return TinyStoriesModel(config)

    def test_forward_shape(self, small_model):
        """Logits should have shape (B, T, vocab_size)."""
        x = torch.randint(0, 256, (2, 16))
        logits, loss = small_model(x)
        assert logits.shape == (2, 16, 256)
        assert loss is None

    def test_forward_with_targets(self, small_model):
        """Loss should be a scalar when targets are provided."""
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        logits, loss = small_model(x, y)
        assert logits.shape == (2, 16, 256)
        assert loss is not None
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0  # cross-entropy is always positive

    def test_generate(self, small_model):
        """Generation should produce tokens longer than the input."""
        x = torch.randint(0, 256, (1, 4))
        output = small_model.generate(x, max_new_tokens=10, temperature=1.0, top_k=10)
        assert output.shape[1] > 4
        assert output.shape[1] <= 14  # at most 4 + 10

    def test_weight_tying(self, small_model):
        """LM head weights should be the same object as token embeddings."""
        assert small_model.lm_head.weight is small_model.token_emb.weight

    def test_parameter_count(self, small_model):
        """Model should report reasonable parameter counts."""
        counts = small_model.count_parameters()
        assert counts["total"] > 0
        assert counts["attention"] > 0
        assert counts["ffn"] > 0

    def test_max_length_enforcement(self, small_model):
        """Sequences exceeding context length should raise an error."""
        x = torch.randint(0, 256, (1, 64))  # exceeds context_length=32
        with pytest.raises(AssertionError, match="exceeds max position"):
            small_model(x)


# ============================================================================
# Vocabulary Tests
# ============================================================================


class TestVocabulary:
    """Tests for the curated vocabulary module."""

    def test_nouns_not_empty(self):
        from tinystories.data.vocabulary import NOUNS
        assert len(NOUNS) > 100

    def test_verbs_not_empty(self):
        from tinystories.data.vocabulary import VERBS
        assert len(VERBS) > 100

    def test_adjectives_not_empty(self):
        from tinystories.data.vocabulary import ADJECTIVES
        assert len(ADJECTIVES) > 100

    def test_sample_story_seed(self):
        """Sampling should return valid word/feature combinations."""
        from tinystories.data.vocabulary import sample_story_seed, NOUNS, VERBS, ADJECTIVES
        import random

        rng = random.Random(42)
        noun, verb, adj, features = sample_story_seed(rng)

        assert noun in NOUNS
        assert verb in VERBS
        assert adj in ADJECTIVES
        assert isinstance(features, list)
        assert len(features) <= 3

    def test_build_generation_prompt(self):
        """Prompt should contain the required words."""
        from tinystories.data.vocabulary import build_generation_prompt

        prompt = build_generation_prompt("cat", "run", "happy", ["Dialogue"])
        assert "cat" in prompt
        assert "run" in prompt
        assert "happy" in prompt
        assert "Dialogue" in prompt
        assert "simple words" in prompt.lower()

    def test_prompt_without_features(self):
        """Prompt should work with an empty feature list."""
        from tinystories.data.vocabulary import build_generation_prompt

        prompt = build_generation_prompt("dog", "jump", "big", [])
        assert "dog" in prompt
        assert "features" not in prompt.lower()
