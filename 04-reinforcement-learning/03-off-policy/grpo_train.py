"""
GRPO Training (Group Relative Policy Optimization)
====================================================
DeepSeek-R1 style training. No reward model, no value network.

How it works:
  For each prompt, generate G responses (a "group").
  Score all G responses with a reward function.
  Normalize scores within the group: advantage = (r - mean(r)) / std(r)
  Update the policy to increase probability of high-advantage responses.

Advantages over PPO:
  - No value network (saves ~1/3 of GPU memory)
  - No separate reward model needed at training time
  - Reward function can be any Python callable

Note on classification: GRPO generates from the current policy (technically
on-policy generation), but it is grouped here with DPO because both methods
avoid a live reward-model-in-the-loop — the key teaching distinction.
"""

import json
import re

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "outputs/grpo"

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Device: {device}")

# ── Reward functions ──────────────────────────────────────────────────────────

def length_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Reward responses between 40-120 words.
    Too short = not informative. Too long = likely padding.
    """
    rewards = []
    for text in completions:
        words = len(text.strip().split())
        if words < 20:
            score = words / 20.0 * 0.3      # heavily penalize very short
        elif words <= 120:
            score = 0.3 + (words - 20) / 100.0 * 0.7  # ramp up to 1.0
        else:
            score = max(0.3, 1.0 - (words - 120) / 100.0)  # penalize verbose
        rewards.append(round(score, 3))
    return rewards


def explanation_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Reward responses that contain explanation structures.
    Signals: numbered lists, 'because', 'for example', colons introducing lists.
    """
    patterns = [
        r"\bbecause\b", r"\bfor example\b", r"\bsuch as\b",
        r"\btherefore\b", r"\bwhich means\b", r"\d+\.",  # numbered list
        r"\bfirst\b.{0,50}\bsecond\b",  # sequential structure
    ]
    rewards = []
    for text in completions:
        hits = sum(1 for p in patterns if re.search(p, text.lower()))
        rewards.append(round(min(hits / 3.0, 1.0), 3))
    return rewards


def combined_reward(completions: list[str], **kwargs) -> list[float]:
    """Weighted combination: 60% length quality + 40% explanation structure."""
    l_rewards = length_reward(completions)
    e_rewards = explanation_reward(completions)
    return [round(0.6 * l + 0.4 * e, 3) for l, e in zip(l_rewards, e_rewards)]


# ── Dataset ───────────────────────────────────────────────────────────────────

# GRPO expects a "prompt" column
with open("preference_data.jsonl") as f:
    records = [json.loads(line) for line in f]

dataset = Dataset.from_list([{"prompt": r["prompt"]} for r in records])
print(f"Dataset: {len(dataset)} prompts")

# ── Training ──────────────────────────────────────────────────────────────────

config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=5,
    # GRPO-specific: number of responses generated per prompt
    num_generations=4,      # G in the paper — increase for better signal, costs more memory
    max_completion_length=150,
    bf16=torch.cuda.is_available(),
)

trainer = GRPOTrainer(
    model=MODEL_NAME,
    reward_funcs=combined_reward,   # any Python callable(completions, **kwargs) -> list[float]
    train_dataset=dataset,
    args=config,
)

print("\nStarting GRPO training...")
print(f"Generating {config.num_generations} responses per prompt, scoring relatively.\n")

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
print(f"\nTraining complete. Model saved to {OUTPUT_DIR}/final")
