"""
GPT-Neo style decoder-only transformer for TinyStories.

Implements the full language model architecture used in the paper:
token + positional embeddings → N transformer blocks → layer norm → LM head.

Each transformer block consists of:
  LayerNorm → CausalSelfAttention → residual → LayerNorm → FFN → residual

The architecture follows the pre-norm (GPT-2/GPT-Neo) convention where
layer normalization is applied before each sub-layer rather than after.

Key design choices from the paper:
- Weight tying between token embeddings and LM head (reduces params)
- Pre-norm residual connections for training stability
- GELU activation in the feed-forward network
- Learned positional embeddings (up to context_length positions)

Reference: Section 3 & Table 1 of Eldan & Li (2023), arXiv:2305.07759
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinystories.config import ModelConfig
from tinystories.model.attention import CausalSelfAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network (FFN).

    Two-layer MLP with GELU activation:
        FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

    The inner dimension (d_ff) is typically 4x the model dimension,
    following standard transformer conventions.

    Args:
        d_model: Input and output dimension.
        d_ff: Inner (hidden) dimension of the FFN.
        dropout: Dropout probability on the output.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-norm residual connections.

    Architecture:
        x → LayerNorm → CausalSelfAttention → + x (residual)
          → LayerNorm → FeedForward → + x (residual)

    Pre-norm (applying LayerNorm before the sub-layer) is used for improved
    training stability, following GPT-2 and GPT-Neo conventions.

    Args:
        config: Model configuration specifying dimensions, heads, and dropout.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            attn_dropout=config.attn_dropout,
            residual_dropout=config.dropout,
            max_seq_len=config.max_position_embeddings,
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attn(self.ln1(x))
        # Pre-norm FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class TinyStoriesModel(nn.Module):
    """GPT-Neo style decoder-only transformer language model.

    The full model stacks N TransformerBlocks between embeddings and
    a language modeling head:

        Input tokens
            → Token Embedding + Position Embedding
            → Dropout
            → TransformerBlock × N
            → LayerNorm
            → LM Head (linear projection to vocab logits)

    Weight tying: The LM head shares weights with the token embedding
    matrix, which is a common technique that reduces parameter count
    and often improves performance.

    Args:
        config: ModelConfig specifying the complete architecture.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm (pre-norm convention: one more LN before the head)
        self.ln_f = nn.LayerNorm(config.d_model)

        # Language modeling head (projects hidden states to vocabulary logits)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share token embedding weights with LM head
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_params_no_emb = sum(
            p.numel() for n, p in self.named_parameters()
            if "token_emb" not in n and "pos_emb" not in n
        )
        print(f"TinyStoriesModel: {n_params:,} parameters "
              f"({n_params_no_emb:,} non-embedding)")

    def _init_weights(self, module: nn.Module):
        """Initialize model weights following GPT-2 conventions.

        - Linear layers and embeddings: N(0, 0.02)
        - Biases: zero
        - LayerNorm: weight=1, bias=0

        The output projections in attention and FFN are scaled by
        1/sqrt(2*n_layers) to stabilize the residual stream.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            # Scale down residual projections
            if hasattr(module, "_is_residual"):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> tuple:
        """Forward pass through the language model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            targets: Optional target token IDs for computing loss.
                     Shape (batch_size, seq_len), shifted by one position
                     relative to input_ids for next-token prediction.

        Returns:
            Tuple of (logits, loss):
            - logits: Shape (batch_size, seq_len, vocab_size)
            - loss: Scalar cross-entropy loss if targets provided, else None
        """
        B, T = input_ids.size()
        device = input_ids.device

        assert T <= self.config.max_position_embeddings, (
            f"Sequence length {T} exceeds max position embeddings "
            f"{self.config.max_position_embeddings}"
        )

        # Create position indices
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        # Embeddings: token + position
        tok_emb = self.token_emb(input_ids)  # (B, T, d_model)
        pos_emb = self.pos_emb(pos)  # (1, T, d_model)
        x = self.emb_dropout(tok_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Language modeling head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # Ignore padding tokens
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Auto-regressive text generation with sampling.

        Generates tokens one at a time, appending each sampled token
        to the input sequence until max_new_tokens is reached or an
        end-of-text token is produced.

        Args:
            input_ids: Starting token IDs, shape (batch_size, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: Keep only top-k highest probability tokens.
            top_p: Nucleus sampling threshold.

        Returns:
            Extended token sequence including generated tokens.
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop input to max context length
            idx_cond = input_ids
            if input_ids.size(1) > self.config.context_length:
                idx_cond = input_ids[:, -self.config.context_length:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get logits for the last position
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                kth_vals = torch.topk(logits, top_k_val, dim=-1).values[:, -1:]
                logits[logits < kth_vals] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[:, 0] = False
                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop on end-of-text token (GPT-2 EOT = 50256)
            if next_token.item() == 50256:
                break

        return input_ids

    def count_parameters(self) -> dict:
        """Count parameters by component for analysis.

        Returns a dictionary breaking down parameter counts by
        embeddings, attention, FFN, and other components.
        """
        counts = {
            "token_embedding": 0,
            "position_embedding": 0,
            "attention": 0,
            "ffn": 0,
            "layer_norm": 0,
            "lm_head": 0,
            "total": 0,
        }

        for name, param in self.named_parameters():
            n = param.numel()
            counts["total"] += n

            if "token_emb" in name:
                counts["token_embedding"] += n
            elif "pos_emb" in name:
                counts["position_embedding"] += n
            elif "attn" in name or "qkv" in name or "out_proj" in name:
                counts["attention"] += n
            elif "ffn" in name or "fc1" in name or "fc2" in name:
                counts["ffn"] += n
            elif "ln" in name:
                counts["layer_norm"] += n
            elif "lm_head" in name:
                counts["lm_head"] += n

        return counts
