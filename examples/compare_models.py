"""
Example: Compare model sizes from the TinyStories paper.

Creates and inspects all model configurations from Table 1 of the paper,
showing how parameter count scales with architecture choices.
"""

from tinystories.config import MODEL_CONFIGS, get_model_config


def compare_architectures():
    """Print a comparison table of all model configurations."""

    print("=" * 80)
    print("TinyStories Model Configurations (Table 1 from the paper)")
    print("=" * 80)
    print(
        f"{'Config':<12} {'Params':>10} {'Layers':>7} {'d_model':>8} "
        f"{'Heads':>6} {'d_ff':>6} {'d_head':>7} {'Context':>8}"
    )
    print("-" * 80)

    for name in ["tiny-1M", "tiny-3M", "tiny-8M", "tiny-28M", "tiny-33M", "tiny-1L"]:
        cfg = get_model_config(name)
        n_params = cfg.num_parameters()

        if n_params >= 1_000_000:
            param_str = f"{n_params / 1_000_000:.1f}M"
        else:
            param_str = f"{n_params / 1_000:.0f}K"

        print(
            f"{name:<12} {param_str:>10} {cfg.n_layers:>7} {cfg.d_model:>8} "
            f"{cfg.n_heads:>6} {cfg.d_ff:>6} {cfg.d_head:>7} {cfg.context_length:>8}"
        )

    print("=" * 80)
    print()

    # Detailed breakdown for 3M model
    cfg = get_model_config("tiny-3M")
    print("Detailed parameter breakdown for tiny-3M:")
    print(f"  Token embeddings:    {cfg.vocab_size} × {cfg.d_model} = {cfg.vocab_size * cfg.d_model:>10,}")
    print(f"  Position embeddings: {cfg.max_position_embeddings} × {cfg.d_model} = {cfg.max_position_embeddings * cfg.d_model:>10,}")

    attn_per_layer = 4 * cfg.d_model * cfg.d_model
    ffn_per_layer = 2 * cfg.d_model * cfg.d_ff
    print(f"  Attention per layer: 4 × {cfg.d_model}² = {attn_per_layer:>10,}")
    print(f"  FFN per layer:       2 × {cfg.d_model} × {cfg.d_ff} = {ffn_per_layer:>10,}")
    print(f"  Total per layer:     {attn_per_layer + ffn_per_layer:>10,}")
    print(f"  × {cfg.n_layers} layers:         {cfg.n_layers * (attn_per_layer + ffn_per_layer):>10,}")
    print(f"  Estimated total:     {cfg.num_parameters():>10,}")
    print()

    # Key insight from the paper
    print("Key insight from the paper:")
    print("  'Scaling depth (more layers) is often more effective than")
    print("   scaling width (larger hidden dim) for these small models.'")


if __name__ == "__main__":
    compare_architectures()
