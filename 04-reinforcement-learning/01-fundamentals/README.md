# 01 — RL Fundamentals

> No training required. No GPU needed. Run in under 1 minute.

## What you will learn

- What a **reward signal** is in the context of LLM training
- Why RL is needed when supervised fine-tuning (SFT) is not enough
- The three algorithms this module covers and when to use each

## The core idea

Supervised fine-tuning teaches a model to imitate examples.
Reinforcement learning teaches a model to **maximize a reward**.

```
SFT:  model output → compare to gold label → update
RL:   model output → reward signal        → update
```

The reward can be:
- A rule (length, keywords, format compliance)
- A classifier (sentiment, safety, helpfulness)
- Human preference (RLHF — Reinforcement Learning from Human Feedback)

## Files

| File | Purpose |
|---|---|
| `data.jsonl` | 25 prompt/response pairs used across this module |
| `reward_demo.py` | Scores responses with 3 rule-based reward functions |

## Run it

```bash
python reward_demo.py
```

Output: a table showing each response's reward score across three dimensions.

## Three reward functions

| Function | What it measures |
|---|---|
| `length_reward` | How detailed the response is (normalized at 80 words) |
| `keyword_reward` | Presence of explanation markers ("because", "for example") |
| `specificity_reward` | Numbers and proper nouns (signals factual grounding) |
| `combined_reward` | Weighted average of the three |

## Why not just use SFT?

SFT requires gold-label responses for every training example. It teaches the model to copy, not to reason about quality. RL allows you to specify **what makes a good response** (via a reward function) and let the model discover how to achieve it — even producing outputs better than any example in your training data.

## Three algorithms, two families

```
RL for LLMs
├── On-policy (02-on-policy/)
│   └── PPO — generates data with current policy, needs live reward model
└── Off-policy (03-off-policy/)
    ├── DPO — learns from preference pairs, no reward model at training time
    └── GRPO — generates groups, scores relatively, no value network needed
```

See `docs/02-on-vs-off-policy.md` for a full comparison.

## Next step

→ [02-on-policy/](../02-on-policy/) — Train with PPO
