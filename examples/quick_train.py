"""
Example: Quick-start training script for TinyStories.

Demonstrates how to programmatically configure and train a small
TinyStories model. This is useful for notebooks, custom experiments,
or when you want more control than the CLI provides.

Requires:
    pip install -e .
    python -m tinystories.data.prepare --output_dir data/tinystories --max_stories 10000
"""

from tinystories.config import ModelConfig, TrainConfig, get_model_config
from tinystories.train import train


def train_tiny_model():
    """Train a minimal model for quick experimentation."""

    # Use the 1M parameter preset or customize your own
    model_config = get_model_config("tiny-1M")

    # Override with custom training settings
    train_config = TrainConfig(
        data_dir="data/tinystories",
        output_dir="checkpoints/quick-experiment",
        learning_rate=1e-3,
        batch_size=64,
        gradient_accumulation_steps=2,
        max_epochs=3,
        eval_every_steps=200,
        save_every_steps=500,
        log_every_steps=25,
        warmup_steps=100,
        seed=42,
        device="auto",
    )

    print(f"Model: {model_config.num_parameters():,} parameters")
    print(f"Architecture: {model_config.n_layers}L / {model_config.d_model}D / {model_config.n_heads}H")
    print(f"Training for {train_config.max_epochs} epochs")
    print()

    model = train(model_config, train_config)
    print("Training complete!")

    return model


if __name__ == "__main__":
    train_tiny_model()
