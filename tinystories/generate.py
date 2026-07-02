"""
Text generation / inference for TinyStories models.

Loads a trained checkpoint and generates stories using auto-regressive
sampling with top-k, top-p (nucleus), and temperature controls. Supports
both single prompts and batch generation from a prompt file.

The paper uses generation to evaluate whether small models can produce
fluent, coherent, multi-paragraph stories that demonstrate reasoning
capabilities despite having far fewer parameters than typical LMs.

Usage:
    # Single prompt
    python -m tinystories.generate \
        --checkpoint checkpoints/tiny-3M/best.pt \
        --prompt "Once upon a time" \
        --max_tokens 200

    # Batch from file
    python -m tinystories.generate \
        --checkpoint checkpoints/tiny-3M/best.pt \
        --prompt_file prompts.txt \
        --num_completions 5

Reference: Section 4 of Eldan & Li (2023), arXiv:2305.07759
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch

from tinystories.config import ModelConfig
from tinystories.model.transformer import TinyStoriesModel

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device = None,
) -> tuple:
    """Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint.
        device: Target device. Auto-detected if None.

    Returns:
        Tuple of (model, tokenizer, config).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model config
    model_config = ModelConfig(**checkpoint["model_config"])

    # Build model and load weights
    model = TinyStoriesModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    logger.info(
        f"Loaded model from {checkpoint_path} "
        f"(step={checkpoint.get('step', '?')}, "
        f"val_loss={checkpoint.get('val_loss', '?')})"
    )

    return model, tokenizer, model_config


def generate_story(
    model: TinyStoriesModel,
    tokenizer,
    prompt: str = "Once upon a time",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device = None,
) -> str:
    """Generate a single story from a prompt.

    Args:
        model: Trained TinyStories model.
        tokenizer: tiktoken encoding instance.
        prompt: Text prompt to start generation from.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (lower = more deterministic).
        top_k: Keep only top-k probability tokens for sampling.
        top_p: Nucleus sampling probability threshold.
        device: Device for computation.

    Returns:
        Generated text including the original prompt.
    """
    if device is None:
        device = next(model.parameters()).device

    # Tokenize the prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    # Decode
    generated_tokens = output_ids[0].tolist()
    text = tokenizer.decode(generated_tokens)

    return text


def generate_batch(
    model: TinyStoriesModel,
    tokenizer,
    prompts: List[str],
    num_completions: int = 1,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device = None,
) -> List[dict]:
    """Generate multiple story completions for a list of prompts.

    For evaluation, the paper generates 10 completions per prompt at
    temperature=1.0 and averages the GPT-4 scores across completions.

    Args:
        model: Trained TinyStories model.
        tokenizer: tiktoken encoding instance.
        prompts: List of text prompts.
        num_completions: Number of completions per prompt.
        max_new_tokens: Maximum tokens to generate per completion.
        temperature: Sampling temperature.
        top_k: Top-k filtering parameter.
        top_p: Nucleus sampling threshold.
        device: Device for computation.

    Returns:
        List of result dicts with prompt, completions, and metadata.
    """
    results = []

    for i, prompt in enumerate(prompts):
        completions = []
        for j in range(num_completions):
            text = generate_story(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
            )
            completions.append(text)
            logger.info(
                f"Prompt {i+1}/{len(prompts)}, "
                f"completion {j+1}/{num_completions}: "
                f"{text[:80]}..."
            )

        results.append({
            "prompt": prompt,
            "completions": completions,
            "params": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
        })

    return results


# ============================================================================
# Default evaluation prompts from the paper
# ============================================================================

DEFAULT_PROMPTS = [
    "Once upon a time",
    "Once upon a time, there was a little girl named Lily.",
    "One day, a boy named Tom went to the park.",
    "There was a big, friendly dog named Max.",
    "In a small village, there lived a kind old woman.",
    "A little bird sat on a tree and sang a happy song.",
    "Once upon a time, there was a small cat who loved to play.",
    "Tom and his sister Lily went to the beach.",
    "One sunny morning, a little rabbit hopped out of its hole.",
    "There was a beautiful garden with many colorful flowers.",
]


def main():
    parser = argparse.ArgumentParser(description="Generate stories with a TinyStories model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--prompt_file", type=str, default=None,
        help="File with prompts (one per line)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=200,
        help="Maximum new tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Top-k filtering (default: 50)",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="Nucleus sampling threshold (default: 0.9)",
    )
    parser.add_argument(
        "--num_completions", type=int, default=1,
        help="Number of completions per prompt (default: 1)",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto/cuda/cpu/mps",
    )
    parser.add_argument(
        "--use_defaults", action="store_true",
        help="Use default evaluation prompts from the paper",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible generation",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Load model
    model, tokenizer, config = load_model_from_checkpoint(args.checkpoint, device)

    # Collect prompts
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.use_defaults:
        prompts = DEFAULT_PROMPTS
    else:
        prompts = ["Once upon a time"]

    # Generate
    if len(prompts) == 1 and args.num_completions == 1:
        # Simple single generation: print directly
        text = generate_story(
            model, tokenizer, prompts[0],
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        print("\n" + "=" * 60)
        print(f"Prompt: {prompts[0]}")
        print("=" * 60)
        print(text)
        print("=" * 60 + "\n")
    else:
        # Batch generation
        results = generate_batch(
            model, tokenizer, prompts,
            num_completions=args.num_completions,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )

        # Print results
        for result in results:
            print("\n" + "=" * 60)
            print(f"Prompt: {result['prompt']}")
            print("-" * 60)
            for i, completion in enumerate(result["completions"]):
                print(f"\n--- Completion {i+1} ---")
                print(completion)
            print("=" * 60)

        # Save if requested
        if args.output_file:
            out_path = Path(args.output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
