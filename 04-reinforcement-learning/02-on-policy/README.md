# 02 — On-Policy Training: PPO

PPO (Proximal Policy Optimization) is the classic RLHF algorithm behind the original ChatGPT alignment work.

## What you will learn

- What "on-policy" means and why it matters
- How the PPO training loop works (generate → reward → KL penalty → update)
- What four components PPO needs and why each exists
- How to use `trl.experimental.ppo.PPOTrainer`

## What is on-policy?

> The policy that **generates** the training data is the **same** policy being updated.

After each update step, old data is stale — the model must generate new responses using its improved weights. This makes PPO sample-inefficient but stable.

## The four PPO components

| Component | Role |
|---|---|
| **Policy model** | The LLM being trained. Generates responses to prompts. |
| **Reference model** | Frozen copy of the initial policy. Provides a KL anchor to prevent catastrophic forgetting. |
| **Value model** | Predicts expected future reward (the "critic"). Shares base weights with the policy. |
| **Reward model** | Scores each generated response with a scalar. |

The training objective balances maximizing reward against staying close to the reference:

```
loss = -reward + beta * KL(policy || reference)
```

## Reward model

This example uses `distilbert-base-uncased-finetuned-sst-2-english` as the reward model — it rewards responses with a confident, positive tone. This is intentionally simple to keep the focus on the training loop, not reward model design.

## Files

| File | Purpose |
|---|---|
| `reward_model.py` | `SentimentRewardModel` — DistilBERT SST-2 wrapper returning scalar reward |
| `ppo_train.py` | Full PPO training loop using `trl.experimental.ppo.PPOTrainer` |
| `inference.py` | Compare base model vs PPO-trained model side-by-side |

## Run it

```bash
# 1. Train
python ppo_train.py

# 2. Compare outputs
python inference.py
```

## Hardware notes

PPO loads four model instances. Estimated VRAM/RAM:

| Hardware | Expected |
|---|---|
| Apple M1/M2 (16GB) | ~14GB RAM, ~10 min for 25 steps |
| NVIDIA T4 (16GB) | ~12GB VRAM, ~5 min |
| CPU only | Works but slow (~30-60 min) |

> **Note:** `trl.experimental.ppo` is the current location of `PPOTrainer` as of TRL 0.9+. It was moved from `trl.trainer` and will eventually be stabilized. Check the [TRL changelog](https://github.com/huggingface/trl/releases) for updates.

## When to use PPO

- You have a reliable reward model or can build one
- You want the most stable, well-understood RL algorithm
- You have GPU memory to spare (4× model size)
- You need fine-grained per-token credit assignment

## Next step

→ [03-off-policy/](../03-off-policy/) — Learn DPO and GRPO (no reward model at training time)
