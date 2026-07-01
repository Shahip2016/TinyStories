"""
Rollout generation and advantage estimation for PPO.

Handles the "experience collection" phase of PPO training:
1. Generate text from the policy given a batch of prompts
2. Score each generation with the reward function
3. Compute per-token log-probabilities under the current and reference policies
4. Estimate advantages using Generalized Advantage Estimation (GAE)

The ``RolloutBuffer`` stores all trajectory data needed for the PPO update
phase (multiple epochs of mini-batch gradient steps).

Reference: Schulman et al. (2017), "Proximal Policy Optimization Algorithms"
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from tinystories.config import ModelConfig, RLConfig

logger = logging.getLogger(__name__)


@dataclass
class RolloutBuffer:
    """Storage for PPO rollout trajectories.

    Holds all data produced during one rollout phase, organized per-sequence.
    After generation and reward computation, the buffer is used for multiple
    PPO optimization epochs.

    All tensors are on the same device and have a batch dimension first.

    Attributes:
        query_ids: Prompt token IDs, shape (batch, prompt_len).
        response_ids: Generated response token IDs, shape (batch, gen_len).
        full_ids: Concatenated query + response IDs, shape (batch, total_len).
        logprobs: Log-probabilities of response tokens under the current policy,
                  shape (batch, gen_len).
        ref_logprobs: Log-probabilities under the reference (frozen) policy,
                      shape (batch, gen_len).
        values: Value estimates at each response position, shape (batch, gen_len).
        rewards: Scalar reward for each sequence, shape (batch,).
        advantages: GAE advantages at each response position, shape (batch, gen_len).
        returns: Discounted returns (advantages + values), shape (batch, gen_len).
        prompts: Original prompt strings.
    """

    query_ids: Optional[torch.Tensor] = None
    response_ids: Optional[torch.Tensor] = None
    full_ids: Optional[torch.Tensor] = None
    logprobs: Optional[torch.Tensor] = None
    ref_logprobs: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    prompts: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        if self.full_ids is not None:
            return self.full_ids.size(0)
        return 0


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple:
    """Compute Generalized Advantage Estimation (GAE-λ).

    GAE provides a family of advantage estimators parameterized by λ ∈ [0, 1]:
    - λ = 0: one-step TD advantage (low variance, high bias)
    - λ = 1: Monte Carlo advantage (high variance, low bias)

    For story generation with a single terminal reward, γ = 1.0 is typical
    since there is no intermediate discounting needed.

    The per-token rewards are constructed by placing the full sequence reward
    at the last token position and zero elsewhere (sparse reward signal).

    Args:
        rewards: Per-sequence scalar rewards, shape (batch,).
        values: Per-position value estimates, shape (batch, gen_len).
        gamma: Discount factor.
        lam: GAE λ parameter.

    Returns:
        Tuple of (advantages, returns):
        - advantages: Shape (batch, gen_len), normalized to zero mean and unit variance.
        - returns: Shape (batch, gen_len), advantages + values.
    """
    batch_size, gen_len = values.shape
    device = values.device

    # Construct per-token rewards: place the scalar reward at the last position
    token_rewards = torch.zeros(batch_size, gen_len, device=device)
    token_rewards[:, -1] = rewards

    # GAE computation (backward pass)
    advantages = torch.zeros(batch_size, gen_len, device=device)
    last_gae = torch.zeros(batch_size, device=device)

    for t in reversed(range(gen_len)):
        if t == gen_len - 1:
            next_value = torch.zeros(batch_size, device=device)
        else:
            next_value = values[:, t + 1]

        delta = token_rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae

    # Returns = advantages + values (used as value function targets)
    returns = advantages + values

    # Normalize advantages (important for PPO stability)
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    if adv_std > 1e-8:
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    return advantages, returns


def compute_logprobs_from_logits(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Extract log-probabilities of specific tokens from logit distributions.

    For each position, computes log P(token_id | context) by applying
    log-softmax to the logits and gathering the target token's probability.

    Args:
        logits: Model output logits, shape (batch, seq_len, vocab_size).
        token_ids: Token IDs to compute log-probs for, shape (batch, seq_len).

    Returns:
        Log-probabilities, shape (batch, seq_len).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather the log-prob for each target token
    token_logprobs = log_probs.gather(
        dim=-1, index=token_ids.unsqueeze(-1)
    ).squeeze(-1)
    return token_logprobs


@torch.no_grad()
def generate_rollouts(
    policy_model,
    ref_model,
    reward_fn: Callable,
    tokenizer,
    prompts: List[str],
    rl_config: RLConfig,
    device: torch.device,
) -> RolloutBuffer:
    """Generate a batch of rollouts for PPO training.

    Pipeline:
    1. Encode prompts and generate responses from the current policy
    2. Score each (prompt, response) pair using the reward function
    3. Compute per-token log-probs under both current and reference policies
    4. Compute GAE advantages and returns

    Args:
        policy_model: The PolicyWithValueHead being trained.
        ref_model: Frozen reference model (TinyStoriesModel or PolicyWithValueHead).
        reward_fn: Function mapping (input_ids, prompts) → reward tensor.
        tokenizer: Tokenizer for encoding prompts.
        prompts: List of prompt strings.
        rl_config: RL training configuration.
        device: Device for computation.

    Returns:
        Populated RolloutBuffer with all trajectory data.
    """
    policy_model.eval()
    buffer = RolloutBuffer(prompts=prompts)

    # Encode prompts
    encoded_prompts = [tokenizer.encode(p) for p in prompts]
    max_prompt_len = max(len(p) for p in encoded_prompts)

    # Pad prompts to same length (left-pad for decoder-only models)
    padded_queries = []
    for tokens in encoded_prompts:
        padding = [tokenizer.encode(" ")[0]] * (max_prompt_len - len(tokens))
        padded_queries.append(padding + tokens)

    query_ids = torch.tensor(padded_queries, dtype=torch.long, device=device)
    buffer.query_ids = query_ids

    # Generate responses from the policy
    generated_ids = policy_model.generate(
        query_ids,
        max_new_tokens=rl_config.max_gen_len,
        temperature=1.0,  # Sample at temperature=1 for RL
        top_k=50,
        top_p=0.95,
    )

    # Split into query and response portions
    prompt_len = query_ids.size(1)
    response_ids = generated_ids[:, prompt_len:]

    # Handle variable-length generations: pad to max gen length
    gen_len = response_ids.size(1)
    if gen_len == 0:
        # Edge case: no tokens generated
        logger.warning("No tokens generated in rollout. Returning empty buffer.")
        return buffer

    buffer.response_ids = response_ids
    buffer.full_ids = generated_ids

    # Compute rewards
    rewards = reward_fn(generated_ids, prompts)
    buffer.rewards = rewards

    # Compute log-probs and values under the current policy
    # Forward pass on full sequence (query + response)
    logits, values, _ = policy_model(generated_ids)

    # Extract response-portion logits and values
    # logits[:, t] predicts token at position t+1, so we use logits[:, prompt_len-1:-1]
    response_logits = logits[:, prompt_len - 1:-1, :]  # (B, gen_len, vocab)
    response_values = values[:, prompt_len - 1:-1]  # (B, gen_len)

    # Clamp gen_len to match
    actual_gen_len = min(response_logits.size(1), response_ids.size(1))
    response_logits = response_logits[:, :actual_gen_len, :]
    response_values = response_values[:, :actual_gen_len]
    response_ids_trimmed = response_ids[:, :actual_gen_len]

    buffer.logprobs = compute_logprobs_from_logits(response_logits, response_ids_trimmed)
    buffer.values = response_values

    # Compute log-probs under the reference policy
    if hasattr(ref_model, "forward"):
        # Check if ref_model returns (logits, values, loss) or (logits, loss)
        ref_output = ref_model(generated_ids)
        if len(ref_output) == 3:
            ref_logits = ref_output[0]
        else:
            ref_logits = ref_output[0]
    else:
        ref_logits, _ = ref_model(generated_ids)

    ref_response_logits = ref_logits[:, prompt_len - 1:-1, :]
    ref_response_logits = ref_response_logits[:, :actual_gen_len, :]
    buffer.ref_logprobs = compute_logprobs_from_logits(
        ref_response_logits, response_ids_trimmed
    )

    # Compute GAE advantages
    advantages, returns = compute_gae(
        rewards=rewards,
        values=response_values.detach(),
        gamma=rl_config.gamma,
        lam=rl_config.lam,
    )
    buffer.advantages = advantages
    buffer.returns = returns

    policy_model.train()
    return buffer
