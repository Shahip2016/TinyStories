"""
Multi-head causal self-attention for the TinyStories transformer.

Implements the standard scaled dot-product attention with a causal mask,
following the GPT-Neo architecture used in the paper. This module handles
query/key/value projections, attention score computation, and output projection.

Reference: Section 3 of Eldan & Li (2023), arXiv:2305.07759
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention.

    Standard transformer attention with a lower-triangular causal mask
    that prevents tokens from attending to future positions. This is the
    core building block of each transformer layer.

    The implementation uses a single fused linear projection for Q, K, V
    (3 * d_model) for efficiency, followed by reshaping into heads.

    Args:
        d_model: Total model hidden dimension.
        n_heads: Number of attention heads.
        attn_dropout: Dropout probability on attention weights.
        residual_dropout: Dropout probability on the output projection.
        max_seq_len: Maximum sequence length for the causal mask buffer.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Fused QKV projection: projects input into Q, K, V simultaneously
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

        # Causal mask: lower-triangular matrix registered as a buffer
        # so it moves with the model to the correct device
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer(
            "causal_mask",
            causal_mask.view(1, 1, max_seq_len, max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of same shape (batch_size, seq_len, d_model).
        """
        B, T, C = x.size()

        # Compute Q, K, V via fused projection and split
        qkv = self.qkv_proj(x)  # (B, T, 3 * d_model)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape into (B, n_heads, T, d_head) for multi-head attention
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention: attn = softmax(Q @ K^T / sqrt(d_k)) @ V
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, T, T)

        # Apply causal mask: set future positions to -inf before softmax
        attn_weights = attn_weights.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        out = attn_weights @ v  # (B, nh, T, d_head)

        # Reshape back: (B, nh, T, d_head) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + dropout
        out = self.residual_dropout(self.out_proj(out))

        return out
