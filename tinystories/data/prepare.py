"""
Dataset preparation: download, tokenize, and shard the TinyStories dataset.

Downloads the TinyStories dataset from HuggingFace (roneneldan/TinyStories),
tokenizes it using the GPT-2 tokenizer, and saves it as memory-mapped binary
shards for efficient training.

Usage:
    python -m tinystories.data.prepare --output_dir data/tinystories
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Special tokens
BOS_TOKEN = "<|endoftext|>"  # GPT-2 uses this as both BOS and EOS


def get_tokenizer():
    """Load the GPT-2 tokenizer via tiktoken.

    The paper uses the GPT-Neo/GPT-2 tokenizer with vocab_size=50257.
    We use tiktoken for fast, efficient tokenization.
    """
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def tokenize_story(tokenizer, text: str, context_length: int = 512) -> list:
    """Tokenize a single story, adding BOS/EOS delimiters.

    Each story is wrapped with the endoftext token and truncated to
    context_length to fit within the model's positional embedding range.

    Args:
        tokenizer: tiktoken encoding instance.
        text: Raw story text.
        context_length: Maximum number of tokens (including special tokens).

    Returns:
        List of token IDs.
    """
    eot = tokenizer.eot_token  # <|endoftext|> token ID
    tokens = [eot] + tokenizer.encode_ordinary(text.strip())

    # Truncate to context length (leaving room for padding if needed)
    if len(tokens) > context_length:
        tokens = tokens[:context_length]

    return tokens


def download_and_prepare(
    output_dir: str = "data/tinystories",
    context_length: int = 512,
    shard_size: int = 100_000,
    dataset_name: str = "roneneldan/TinyStories",
    split: Optional[str] = None,
    max_stories: Optional[int] = None,
):
    """Download TinyStories from HuggingFace and create tokenized shards.

    The dataset is tokenized and packed into fixed-length binary shards
    (.bin files) containing uint16 token arrays. This format enables
    memory-mapped loading during training for maximum throughput.

    Args:
        output_dir: Directory to save tokenized shards.
        context_length: Maximum sequence length for tokenization.
        shard_size: Number of tokens per shard file.
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to use (None = all available splits).
        max_stories: Maximum number of stories to process (None = all).
    """
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset: {dataset_name}")

    # Load dataset
    if split:
        dataset = load_dataset(dataset_name, split=split)
        splits = {split: dataset}
    else:
        dataset = load_dataset(dataset_name)
        splits = dataset

    tokenizer = get_tokenizer()

    for split_name, split_data in splits.items():
        logger.info(f"Processing split: {split_name} ({len(split_data)} stories)")

        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Tokenize all stories
        all_tokens = []
        n_stories = min(len(split_data), max_stories) if max_stories else len(split_data)

        for i in tqdm(range(n_stories), desc=f"Tokenizing {split_name}"):
            text = split_data[i]["text"]
            if text and text.strip():
                tokens = tokenize_story(tokenizer, text, context_length)
                all_tokens.extend(tokens)

        logger.info(
            f"Split '{split_name}': {n_stories} stories -> "
            f"{len(all_tokens):,} tokens"
        )

        # Write shards
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        n_shards = (len(all_tokens) + shard_size - 1) // shard_size

        for shard_idx in range(n_shards):
            start = shard_idx * shard_size
            end = min(start + shard_size, len(all_tokens))
            shard = all_tokens[start:end]

            shard_path = split_dir / f"shard_{shard_idx:05d}.bin"
            shard.tofile(str(shard_path))

        logger.info(f"Wrote {n_shards} shards to {split_dir}")

    # Save metadata
    meta = {
        "dataset": dataset_name,
        "context_length": context_length,
        "shard_size": shard_size,
        "vocab_size": tokenizer.n_vocab,
        "eot_token": tokenizer.eot_token,
    }

    import json
    with open(output_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Dataset preparation complete. Output: {output_path}")


class TinyStoriesDataset:
    """Memory-mapped dataset loader for tokenized TinyStories shards.

    Loads pre-tokenized binary shards and serves fixed-length sequences
    for next-token prediction training. Supports random access via
    memory mapping for efficient data loading without loading the entire
    dataset into RAM.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        context_length: int = 512,
    ):
        """Initialize the dataset from tokenized shards.

        Args:
            data_dir: Root directory containing tokenized shards.
            split: Dataset split ('train' or 'validation').
            context_length: Number of tokens per training example.
        """
        self.context_length = context_length
        self.split = split

        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                f"Run 'python -m tinystories.data.prepare' first."
            )

        # Load all shards into a single contiguous array via memory mapping
        shard_files = sorted(split_dir.glob("shard_*.bin"))
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {split_dir}")

        logger.info(f"Loading {len(shard_files)} shards from {split_dir}")

        # Memory-map all shards
        shards = []
        for sf in shard_files:
            shard = np.memmap(str(sf), dtype=np.uint16, mode="r")
            shards.append(shard)

        self.data = np.concatenate(shards)
        self.n_tokens = len(self.data)
        self.n_examples = self.n_tokens // context_length

        logger.info(
            f"Loaded {self.n_tokens:,} tokens ({self.n_examples:,} examples) "
            f"from {split} split"
        )

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int):
        """Get a single training example (input, target) pair.

        For next-token prediction: input = tokens[i:i+L], target = tokens[i+1:i+L+1]
        """
        import torch

        start = idx * self.context_length
        end = start + self.context_length + 1

        # Ensure we don't go out of bounds
        if end > self.n_tokens:
            start = self.n_tokens - self.context_length - 1
            end = self.n_tokens

        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare the TinyStories dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/tinystories",
        help="Output directory for tokenized shards",
    )
    parser.add_argument(
        "--context_length", type=int, default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--shard_size", type=int, default=100_000,
        help="Tokens per shard file (default: 100000)",
    )
    parser.add_argument(
        "--max_stories", type=int, default=None,
        help="Maximum stories to process (default: all)",
    )
    parser.add_argument(
        "--dataset", type=str, default="roneneldan/TinyStories",
        help="HuggingFace dataset name",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    download_and_prepare(
        output_dir=args.output_dir,
        context_length=args.context_length,
        shard_size=args.shard_size,
        dataset_name=args.dataset,
        max_stories=args.max_stories,
    )


if __name__ == "__main__":
    main()
