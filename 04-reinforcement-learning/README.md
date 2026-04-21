# 04 — Reinforcement Learning for LLM Alignment

> Standalone module. No dependencies on the fine-tuning track (01–03).

This module teaches modern RL techniques for aligning language models: PPO (on-policy), DPO and GRPO (off-policy), and how to run all three via LlamaFactory's YAML interface.

## Quick Start

```bash
# No training — understand reward signals in 2 minutes
cd 01-fundamentals && python reward_demo.py

# On-policy PPO (~10-30 min depending on hardware)
cd 02-on-policy && python ppo_train.py

# Off-policy DPO (~5-10 min)
cd 03-off-policy && python dpo_train.py

# Off-policy GRPO (~8-15 min)
cd 03-off-policy && python grpo_train.py

# Same training, zero Python (LlamaFactory)
cd appendix-llamafactory && ./run.sh dpo
```

## Module Structure

```
04-reinforcement-learning/
├── 01-fundamentals/     What is a reward? Why RL? (no training)
├── 02-on-policy/        PPO — classic RLHF, live reward model
├── 03-off-policy/       DPO + GRPO — preference data, no live reward model
├── docs/                Concept deep-dives
└── appendix-llamafactory/ Same training via YAML + CLI
```

## The Central Question

> **On-policy or off-policy?**

```
On-policy (PPO)
  ✓ Most stable, fine-grained credit assignment
  ✗ Needs live reward model + value network (4× memory)

Off-policy (DPO)
  ✓ Simplest, only needs preference pairs
  ✗ Bounded by quality of preference data

Off-policy style (GRPO)
  ✓ No reward model, no value network (2× memory)
  ✓ Can exceed human preferences via automated reward
  ✗ Needs a clear, measurable reward function
```

See `docs/02-on-vs-off-policy.md` for the full comparison.

## Model

All scripts use `Qwen/Qwen2.5-0.5B-Instruct`. Hardware requirements:

| Method | Memory | Time (M2 16GB) |
|---|---|---|
| 01 fundamentals | < 1GB | < 1 min |
| 02 PPO | ~14GB RAM | ~10-30 min |
| 03 DPO | ~6GB RAM | ~5-10 min |
| 03 GRPO | ~8GB RAM | ~8-15 min |

Runs on Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU (slow).

## Prerequisites

```bash
pip install "trl>=0.8.0" accelerate transformers datasets torch
```

For the LlamaFactory appendix:
```bash
pip install llamafactory
```

## Concept Docs

- [RL Fundamentals](docs/01-rl-fundamentals.md) — reward, policy, KL divergence
- [On-Policy vs Off-Policy](docs/02-on-vs-off-policy.md) — the core distinction
- [Algorithm Comparison](docs/03-algorithm-comparison.md) — PPO vs DPO vs GRPO table

## Relation to this Repo

This module is standalone and self-contained. The fine-tuning modules (01–03) covered supervised learning (LoRA, full fine-tuning). This module covers reinforcement learning — the next step in the standard alignment pipeline: **Pretrain → SFT → RL**.
