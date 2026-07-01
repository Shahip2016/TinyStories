# TinyStories

**Implementation of ["TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"](https://arxiv.org/abs/2305.07759)**  
*Ronen Eldan & Yuanzhi Li, 2023*

---

## Overview

This repository provides a complete, from-scratch implementation of the TinyStories paper. The paper demonstrates that language models with **fewer than 10 million parameters** can produce fluent, grammatically correct, and coherent short stories when trained on a synthetic dataset of simple children's stories.

### Key Contributions Implemented

| Component | Description |
|---|---|
| **Synthetic Dataset Generation** | GPT-3.5/GPT-4-powered pipeline to generate stories constrained to a 3–4 year old's vocabulary |
| **Small Transformer Models** | GPT-Neo-style decoder-only transformers from 1M to 33M parameters |
| **Training Pipeline** | Full training loop with mixed precision, gradient accumulation, and cosine LR scheduling |
| **RL Fine-Tuning (PPO)** | Post-SFT optimization via Proximal Policy Optimization with learned or heuristic reward |
| **GPT-Eval Framework** | GPT-4-as-judge evaluation scoring grammar, creativity, and consistency (1–10 scale) |
| **Text Generation** | Inference with top-k, top-p, and temperature sampling |

## Project Structure

```
tinystories/
├── config.py              # Model, training & RL configurations (1M → 33M)
├── data/
│   ├── vocabulary.py      # Curated ~1500 word vocabulary for 3-4 year olds
│   ├── generate_stories.py # Synthetic story generation via OpenAI API
│   └── prepare.py         # Download & tokenize HuggingFace dataset
├── model/
│   ├── transformer.py     # GPT-Neo decoder-only transformer
│   └── attention.py       # Multi-head causal self-attention
├── rl/                    # Reinforcement learning fine-tuning
│   ├── reward.py          # Learned & heuristic reward models
│   ├── value.py           # Value head & PolicyWithValueHead wrapper
│   ├── rollout.py         # Rollout buffer & GAE advantage estimation
│   └── ppo_trainer.py     # PPO training loop
├── train.py               # SFT training loop with checkpointing
├── train_rl.py            # RL fine-tuning CLI entry point
├── generate.py            # Story generation / inference
└── evaluate.py            # GPT-4 evaluation framework
```

## Quick Start

### Installation

```bash
pip install -e .
```

### 1. Prepare the Dataset

Download and tokenize the TinyStories dataset from HuggingFace:

```bash
python -m tinystories.data.prepare --output_dir data/tinystories
```

### 2. Train a Model

Train a small (3.6M parameter) model:

```bash
python -m tinystories.train \
    --config tiny-3M \
    --data_dir data/tinystories \
    --output_dir checkpoints/tiny-3M \
    --epochs 10
```

Available configurations: `tiny-1M`, `tiny-3M`, `tiny-8M`, `tiny-28M`, `tiny-33M`

### 3. Generate Stories

```bash
python -m tinystories.generate \
    --checkpoint checkpoints/tiny-3M/best.pt \
    --prompt "Once upon a time" \
    --max_tokens 200 \
    --temperature 0.8
```

### 4. Evaluate with GPT-4

```bash
export OPENAI_API_KEY="your-key-here"
python -m tinystories.evaluate \
    --checkpoint checkpoints/tiny-3M/best.pt \
    --num_prompts 50 \
    --output_file results/eval_tiny3M.json
```

### 5. Generate Synthetic Training Data (Optional)

Generate your own TinyStories-style dataset:

```bash
export OPENAI_API_KEY="your-key-here"
python -m tinystories.data.generate_stories \
    --model gpt-4 \
    --num_stories 1000 \
    --output_file data/custom_stories.json
```

## Model Configurations

| Config | Params | Layers | Hidden Dim | Heads | Context |
|--------|--------|--------|------------|-------|---------|
| `tiny-1M` | ~1M | 4 | 64 | 8 | 512 |
| `tiny-3M` | ~3.6M | 8 | 128 | 16 | 512 |
| `tiny-8M` | ~8M | 8 | 256 | 8 | 512 |
| `tiny-28M` | ~28M | 16 | 512 | 8 | 512 |
| `tiny-33M` | ~33M | 16 | 576 | 12 | 512 |

## RL Fine-Tuning (PPO)

After supervised pre-training, models can be further optimized using **Proximal Policy Optimization** (PPO) against a reward signal. Two reward strategies are supported:

| Strategy | Description | Data Required |
|----------|-------------|---------------|
| **Heuristic** | Rule-based scoring: vocabulary diversity, repetition penalty, sentence structure, length | None |
| **Learned** | Transformer-based reward model trained on quality preferences | Preference labels |

### Fine-tune with heuristic reward (no extra data needed)

```bash
python -m tinystories.train_rl \
    --checkpoint checkpoints/tiny-3M/best.pt \
    --use_heuristic_reward \
    --output_dir checkpoints/rl-tiny-3M \
    --total_steps 500
```

### Fine-tune with a learned reward model

```bash
python -m tinystories.train_rl \
    --checkpoint checkpoints/tiny-3M/best.pt \
    --reward_model checkpoints/reward_model.pt \
    --output_dir checkpoints/rl-tiny-3M
```

Key PPO hyperparameters can be overridden from the CLI: `--lr`, `--clip_range`, `--kl_coeff`, `--ppo_epochs`, `--max_gen_len`.

## Evaluation (GPT-Eval)

The paper introduces GPT-4 as an automated grader. Stories are scored on three dimensions (1–10):

- **Grammar**: Fluency, correctness, and syntactic structure
- **Creativity**: Originality, diversity, and engagement
- **Consistency**: Narrative coherence, character identity, and logical flow

## Citation

```bibtex
@article{eldan2023tinystories,
  title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author={Eldan, Ronen and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2305.07759},
  year={2023}
}
```

## License

MIT
