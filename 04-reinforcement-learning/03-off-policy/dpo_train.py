"""
DPO Training (Direct Preference Optimization)
===============================================
Off-policy method: train directly from preference pairs.
No reward model needed. No live generation during training.

How it works:
  Given (prompt, chosen_response, rejected_response), DPO adjusts
  the model to increase the log-probability of 'chosen' relative to
  'rejected', using the initial model as a reference (implicit reward).

  This is derived from the Bradley-Terry preference model and avoids
  explicit reward modeling entirely.
"""

import json
import torch
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "outputs/dpo"

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Device: {device}")

# ── Dataset ───────────────────────────────────────────────────────────────────

# DPO expects columns: prompt, chosen, rejected
with open("preference_data.jsonl") as f:
    records = [json.loads(line) for line in f]

dataset = Dataset.from_list(records)
print(f"Dataset: {len(dataset)} preference pairs")

# ── Training ──────────────────────────────────────────────────────────────────

config = DPOConfig(
    output_dir=OUTPUT_DIR,
    beta=0.1,               # KL penalty — higher = stay closer to reference model
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=5,
    # Use bf16 on GPU, fp32 on MPS/CPU
    bf16=torch.cuda.is_available(),
)

trainer = DPOTrainer(
    model=MODEL_NAME,
    train_dataset=dataset,
    args=config,
)

print("\nStarting DPO training...")
print("No reward model needed — preference pairs are the training signal.\n")

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
print(f"\nTraining complete. Model saved to {OUTPUT_DIR}/final")
