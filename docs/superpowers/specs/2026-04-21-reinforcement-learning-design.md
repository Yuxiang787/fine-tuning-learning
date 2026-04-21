# Reinforcement Learning Module Design

**Date:** 2026-04-21  
**Status:** Approved  
**Scope:** New standalone folder `04-reinforcement-learning/` added to the fine-tuning-learning teaching repo

---

## Overview

A standalone, teaching-focused reinforcement learning module for LLM alignment. The module starts from first principles (what is a reward signal?) and progressively introduces the two major families of modern LLM RL training: on-policy (PPO) and off-policy (DPO, GRPO). An appendix shows how to run the same algorithms via LlamaFactory's YAML-driven CLI.

The module is **self-contained** — it does not follow the 3-stage progressive structure of the fine-tuning track. RL for LLMs has its own conceptual depth that is best expressed through the on-policy vs off-policy fork, not a beginner/intermediate/advanced split.

---

## Folder Structure

```
04-reinforcement-learning/
├── README.md
│
├── 01-fundamentals/
│   ├── README.md
│   ├── reward_demo.py
│   └── data.jsonl
│
├── 02-on-policy/
│   ├── README.md
│   ├── reward_model.py
│   ├── ppo_train.py
│   └── inference.py
│
├── 03-off-policy/
│   ├── README.md
│   ├── preference_data.jsonl
│   ├── dpo_train.py
│   ├── grpo_train.py
│   └── inference.py
│
├── docs/
│   ├── 01-rl-fundamentals.md
│   ├── 02-on-vs-off-policy.md
│   └── 03-algorithm-comparison.md
│
└── appendix-llamafactory/
    ├── README.md
    ├── configs/
    │   ├── ppo_qwen.yaml
    │   ├── dpo_qwen.yaml
    │   └── grpo_qwen.yaml
    └── run.sh
```

---

## Architecture

### Core Library: TRL (HuggingFace)

Sections 02 and 03 use TRL's high-level trainers:

- `PPOTrainer` — on-policy training with a live reward model
- `DPOTrainer` — off-policy training from preference pairs
- `GRPOTrainer` — off-policy with group-relative policy optimization (DeepSeek-R1 style)

TRL is the standard library for LLM RL training and maps directly to the academic algorithms, making it ideal for teaching.

### Model: Qwen2.5-0.5B

Consistent with the rest of the repo. Runs on:

- Apple Silicon via MPS (`device="mps"`)
- NVIDIA GPU via CUDA (`device="cuda"`)
- CPU fallback (slow but functional)

Each script auto-detects the available device.

### Appendix Library: LlamaFactory

LlamaFactory wraps TRL and other backends behind a YAML config + CLI interface. The appendix shows that the same PPO/DPO/GRPO training can be launched with `llamafactory-cli train config.yaml` — no Python code required. This demonstrates the "framework" layer above raw TRL.

---

## Components

### 01-fundamentals

**Purpose:** Build intuition before any training. Learners should understand what a reward signal is in the LLM context and why RL is needed at all (vs supervised fine-tuning alone).

**`reward_demo.py`**

- Loads `data.jsonl` (20–30 prompt/response pairs)
- Applies a rule-based reward function (e.g., response length, keyword presence, sentiment score)
- Prints scored outputs — no model training, no GPU required
- Teaches: reward is just a scalar score on a model output

**`data.jsonl`**

- 20–30 hand-crafted prompt/response pairs
- Used in 01 for reward demo and in 02-on-policy as prompt seeds
- 03-off-policy uses its own `preference_data.jsonl` with `(prompt, chosen, rejected)` triplets — different format, separate file

**`README.md`** covers:

- What RL for LLMs means (reward, policy, KL penalty)
- Why RLHF exists (SFT alone doesn't capture human preference)
- The three algorithms covered and when to use each

---

### 02-on-policy (PPO)

**Purpose:** Show PPO as the classic RLHF algorithm. Learners see that on-policy training requires a reward model to be active during the training loop — the policy generates outputs, the reward model scores them, the policy updates.

**`reward_model.py`**

- A tiny sentiment-based or rule-based reward model
- Returns a scalar score for any model output
- Kept simple intentionally — the point is the interface, not the model quality

**`ppo_train.py`**

- Loads Qwen2.5-0.5B as the policy model
- Loads the reward model from `reward_model.py`
- Runs PPO via `trl.PPOTrainer`
- Logs reward per step so learners can see learning happening

**`inference.py`**

- Loads both the base model and the PPO-trained checkpoint
- Runs the same prompts through both
- Side-by-side output comparison to show what PPO changed

**`README.md`** covers:

- What "on-policy" means: the policy that generates data is the same policy being updated
- PPO's actor-critic structure (brief, conceptual)
- When to use PPO: when you have a reliable reward signal or reward model

---

### 03-off-policy (DPO + GRPO)

**Purpose:** Show that preference-based methods eliminate the need for a live reward model by learning directly from preference data. DPO and GRPO are presented side by side as two approaches to the same problem.

> **Note on GRPO classification:** GRPO generates responses from the current policy (technically on-policy generation), but it optimizes without a value/critic network using group-relative scores. It is taught here alongside DPO because both avoid the reward-model-in-the-loop requirement of PPO — the key teaching distinction. The README will surface this nuance explicitly.

**`preference_data.jsonl`**

- 20–30 `(prompt, chosen_response, rejected_response)` triplets
- Separate from 01's `data.jsonl` — different schema

**`dpo_train.py`**

- Takes `preference_data.jsonl` as input
- Trains via `trl.DPOTrainer`
- No reward model involved — the preference signal is implicit in the data

**`grpo_train.py`**

- Group Relative Policy Optimization (from DeepSeek-R1)
- Generates a group of responses per prompt, scores them relatively
- Trains via `trl.GRPOTrainer`
- Shows why GRPO reduces variance compared to PPO without a value network

**`inference.py`**

- Compares outputs from the DPO-trained and GRPO-trained checkpoints
- Helps learners see the practical difference between the two methods

**`README.md`** covers:

- What "off-policy" means: the data used for updates was generated by a different (or older) policy
- DPO: preference pairs as implicit reward, derived from Bradley-Terry model
- GRPO: group scoring removes the need for a critic/value function
- When to choose DPO vs GRPO

---

### docs/

Three short conceptual documents:

**`01-rl-fundamentals.md`** — Reward, policy, value function, KL divergence penalty explained in plain language with diagrams.

**`02-on-vs-off-policy.md`** — The central distinction of this module. Covers: data freshness, sample efficiency, stability, infrastructure cost. Includes a decision table.

**`03-algorithm-comparison.md`** — Comparison table: PPO vs DPO vs GRPO across axes of complexity, data requirements, stability, compute cost, and use cases.

---

### appendix-llamafactory/

**Purpose:** Show that production RL training doesn't require writing Python. LlamaFactory's YAML interface covers PPO, DPO, and GRPO with a single config file per method.

**`configs/ppo_qwen.yaml`** — PPO config: model path, dataset, reward model, learning rate, PPO-specific hyperparams.

**`configs/dpo_qwen.yaml`** — DPO config: model path, preference dataset, beta coefficient.

**`configs/grpo_qwen.yaml`** — GRPO config: model path, dataset, group size, reward function reference.

**`run.sh`** — Three commands:

```bash
llamafactory-cli train configs/ppo_qwen.yaml
llamafactory-cli train configs/dpo_qwen.yaml
llamafactory-cli train configs/grpo_qwen.yaml
```

**`README.md`** explicitly contrasts raw TRL vs LlamaFactory:

- Raw TRL: full control, more code, better for research and custom reward functions
- LlamaFactory: production-ready, minimal code, better for standard alignment tasks

---

## Data Flow

```
01-fundamentals
  data.jsonl ──→ reward_demo.py ──→ scored outputs (no training)

02-on-policy
  prompts ──→ policy model ──→ outputs ──→ reward_model ──→ PPO update ──→ trained policy

03-off-policy (DPO)
  (prompt, chosen, rejected) ──→ DPOTrainer ──→ trained policy

03-off-policy (GRPO)
  prompts ──→ generate N responses ──→ relative scoring ──→ GRPOTrainer ──→ trained policy

appendix-llamafactory
  YAML config ──→ llamafactory-cli ──→ same trained policy as above
```

---

## Hardware

| Setup | Support |
|---|---|
| Apple Silicon (M1/M2/M3/M4) | MPS device, auto-detected |
| NVIDIA GPU | CUDA device, auto-detected |
| CPU only | Supported, slow — noted in each script |

Each script includes a device detection block:

```python
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Dependencies

Added to the repo's `requirements.txt`:

```text
trl>=0.8.0
llamafactory  # appendix only
```

---

## Language Conventions

All content in English — folder names, file names, README files, docs, code, and comments.

- **Comments**: minimal, only where the WHY is non-obvious

---

## Learning Arc

1. **01-fundamentals** — understand the RL problem formulation for LLMs without writing a training loop
2. **02-on-policy** — PPO: the classic RLHF approach, requires an active reward model
3. **03-off-policy** — DPO and GRPO: learn from preference data, no reward model needed during training
4. **appendix** — same algorithms, zero Python, one YAML config per method; shows the framework abstraction layer
