# 03 — Off-Policy Training: DPO and GRPO

Both methods here avoid the need for a live reward model during training. DPO learns from human preference pairs; GRPO generates its own comparisons using a reward function.

## What you will learn

- What "off-policy" means for LLM training
- How DPO turns preference pairs into an implicit reward (no reward model at training time)
- How GRPO uses group-relative scoring to avoid a value network
- When to choose DPO vs GRPO

## DPO — Direct Preference Optimization

Given `(prompt, chosen, rejected)` triplets, DPO adjusts the model so that:
- The log-probability of `chosen` increases
- The log-probability of `rejected` decreases

The reference model (initial weights, frozen) acts as an implicit reward anchor, preventing the model from drifting too far. No reward model runs during training.

```
Training data: (prompt, chosen_response, rejected_response)
                              ↓
              DPOTrainer adjusts model weights
                              ↓
         Model more likely to generate chosen-style responses
```

**Key hyperparameter:** `beta` — the KL penalty coefficient. Higher beta = stay closer to the reference model. Typical range: 0.01–0.5.

## GRPO — Group Relative Policy Optimization

GRPO was introduced in DeepSeek-R1. For each training prompt:
1. Generate G responses (a "group") from the current model
2. Score each with a reward function
3. Normalize scores within the group: `advantage = (r - mean) / std`
4. Update model to increase probability of high-advantage responses

```
prompt → generate G responses → score → normalize within group → update
```

**No value network needed** — group normalization replaces the critic that PPO requires.

> **Classification note:** GRPO generates from the current policy (technically on-policy generation), but it is taught here alongside DPO because both avoid the reward-model-in-the-loop design of PPO. This is the key architectural distinction.

## Files

| File | Purpose |
|---|---|
| `preference_data.jsonl` | 25 `(prompt, chosen, rejected)` triplets |
| `dpo_train.py` | DPO training via `trl.DPOTrainer` |
| `grpo_train.py` | GRPO training via `trl.GRPOTrainer` with custom reward function |
| `inference.py` | Compare DPO vs GRPO outputs side-by-side |

## Run it

```bash
# Train with DPO
python dpo_train.py

# Train with GRPO
python grpo_train.py

# Compare outputs
python inference.py
```

## DPO vs GRPO comparison

| Aspect | DPO | GRPO |
|---|---|---|
| Training data | Preference pairs (chosen/rejected) | Prompts only |
| Reward signal | Implicit (from pairs) | Explicit reward function |
| Value network | No | No |
| Data collection | Requires human labeling | Can use automated reward |
| Best for | Aligning to human style preferences | Optimizing measurable objectives |

## Hardware notes

Both methods require ~2× model size in memory (model + reference). Estimated:

| Hardware | DPO | GRPO |
|---|---|---|
| Apple M1/M2 (16GB) | ~6GB RAM, ~5 min | ~8GB RAM, ~8 min |
| NVIDIA T4 (16GB) | ~5GB VRAM, ~3 min | ~6GB VRAM, ~4 min |

## When to choose DPO vs GRPO

- **DPO:** You have high-quality preference data (human-labeled or from a strong judge model). You want to align to a specific style or set of values.
- **GRPO:** You have a clear, measurable reward function (code correctness, math accuracy, format compliance). You want the model to discover strategies that maximize it.

## Next step

→ [appendix-llamafactory/](../appendix-llamafactory/) — Run the same training with zero Python
