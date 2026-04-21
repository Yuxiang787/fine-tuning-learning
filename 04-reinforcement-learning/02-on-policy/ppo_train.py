"""
PPO Training for LLM Alignment
================================
On-policy method: the policy that generates responses IS the policy being updated.

Components:
  - Policy model:    Qwen2.5-0.5B-Instruct (generates responses)
  - Value model:     Same base, adds a value head to estimate expected reward
  - Reference model: Frozen copy of initial policy (KL penalty anchor)
  - Reward model:    SentimentRewardModel (scores each response 0-1)

PPO update loop:
  1. Generate responses with policy model
  2. Score responses with reward model
  3. Compute KL penalty vs reference model
  4. Update policy to maximize: reward - beta * KL
"""

import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer

# PPO has moved to trl.experimental.ppo as of TRL 0.9+
from trl.experimental.ppo import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)

sys.path.insert(0, str(Path(__file__).parent))
from reward_model import load_reward_model, load_reward_tokenizer

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "outputs/ppo"

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Device: {device}")

# ── Dataset ──────────────────────────────────────────────────────────────────

data_path = Path(__file__).parent.parent / "01-fundamentals" / "data.jsonl"
records = [json.loads(line) for line in data_path.read_text().splitlines()]
dataset = Dataset.from_list([{"prompt": r["prompt"]} for r in records])

# ── Models ───────────────────────────────────────────────────────────────────

print("Loading models (this may take a few minutes)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Policy + value model: same base weights, value head added on top
model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)

# Reference model: frozen copy of the initial policy — used for KL penalty
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)

# Value model: predicts expected future reward (critic in actor-critic)
value_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)

# Reward model: scores each generated response
reward_model = load_reward_model(device)

# ── PPO Config ───────────────────────────────────────────────────────────────

config = PPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,
    # Reduce batch size for MPS/CPU; increase for multi-GPU
    per_device_train_batch_size=2 if device in ("mps", "cpu") else 8,
    gradient_accumulation_steps=4,
    num_ppo_epochs=1,
    total_episodes=len(dataset),
    kl_coef=0.05,           # penalize drifting too far from reference policy
    cliprange=0.2,           # standard PPO clip range
    missing_eos_penalty=1.0, # penalize incomplete responses
    stop_token="eos",
    logging_steps=5,
)

# ── Trainer ──────────────────────────────────────────────────────────────────

trainer = PPOTrainer(
    args=config,
    processing_class=tokenizer,
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=dataset,
)

print(f"\nStarting PPO training on {len(dataset)} prompts...")
print("Watch 'objective/rlhf_reward' in logs — it should trend upward.\n")

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
print(f"\nTraining complete. Model saved to {OUTPUT_DIR}/final")
