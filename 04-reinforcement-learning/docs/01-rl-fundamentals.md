# RL Fundamentals for LLMs

## The Four Core Concepts

### 1. Policy (π)

The policy is the LLM itself — the function that maps a prompt to a response.

```
policy(prompt) → response
```

During RL training, we adjust the policy's weights to produce higher-reward responses.

### 2. Reward (r)

A scalar number that scores a response. Higher is better.

```
reward(prompt, response) → float
```

The reward can be:
- **Rule-based:** length, keyword presence, format compliance
- **Model-based:** a classifier (sentiment, safety, helpfulness)
- **Human feedback:** direct human ratings (RLHF)

### 3. KL Divergence Penalty

Without a constraint, RL training can "reward hack" — finding degenerate responses that score high on the reward function but are useless (e.g., repeating high-reward phrases endlessly).

The KL divergence penalty keeps the policy close to a **reference model** (the initial weights):

```
loss = -reward + beta × KL(current_policy || reference_policy)
```

This prevents the model from forgetting its language skills while chasing reward.

### 4. Value Function (PPO only)

The value function estimates the expected future reward from a given state. It acts as a baseline to reduce variance in training updates. This is the "critic" in actor-critic methods.

GRPO eliminates the value function by using group normalization instead.

---

## Why RL, Not SFT?

| Supervised Fine-Tuning | Reinforcement Learning |
|---|---|
| Needs gold-label responses | Needs only a reward signal |
| Teaches model to copy | Teaches model to optimize |
| Bounded by training data quality | Can exceed human-written examples |
| Stable, predictable training | More complex, can be unstable |

**Use SFT first.** RL is most effective on a model that already has strong language capabilities. The standard pipeline is: Pretrain → SFT → RL.

---

## The RLHF Pipeline

```
1. Pretrain LLM on large text corpus
         ↓
2. SFT on curated instruction-following examples
         ↓
3. Collect human preference data (A vs B comparisons)
         ↓
4. Train a reward model on preference data (PPO) or skip (DPO/GRPO)
         ↓
5. RL fine-tuning (PPO / DPO / GRPO)
```

This module covers step 5 (and step 4 for PPO).
