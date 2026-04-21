# Reinforcement Learning Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `04-reinforcement-learning/` teaching module covering RL fundamentals, on-policy PPO, and off-policy DPO+GRPO for LLM alignment, with a LlamaFactory appendix.

**Architecture:** Three progressive sections (fundamentals → on-policy → off-policy) plus a LlamaFactory appendix. Each section is fully self-contained. TRL provides the trainer layer (`trl.experimental.ppo.PPOTrainer`, `trl.DPOTrainer`, `trl.GRPOTrainer`); LlamaFactory provides the YAML-driven CLI abstraction.

**Tech Stack:** Python 3.11, TRL ≥ 0.8, HuggingFace Transformers, Datasets, Qwen/Qwen2.5-0.5B-Instruct, DistilBERT (reward model), LlamaFactory CLI

---

## File Map

```
04-reinforcement-learning/
├── README.md                          # create
├── 01-fundamentals/
│   ├── README.md                      # create
│   ├── data.jsonl                     # create
│   └── reward_demo.py                 # create
├── 02-on-policy/
│   ├── README.md                      # create
│   ├── reward_model.py                # create
│   ├── ppo_train.py                   # create
│   └── inference.py                   # create
├── 03-off-policy/
│   ├── README.md                      # create
│   ├── preference_data.jsonl          # create
│   ├── dpo_train.py                   # create
│   ├── grpo_train.py                  # create
│   └── inference.py                   # create
├── docs/
│   ├── 01-rl-fundamentals.md          # create
│   ├── 02-on-vs-off-policy.md        # create
│   └── 03-algorithm-comparison.md    # create
└── appendix-llamafactory/
    ├── README.md                      # create
    ├── configs/
    │   ├── ppo_qwen.yaml              # create
    │   ├── dpo_qwen.yaml              # create
    │   └── grpo_qwen.yaml             # create
    └── run.sh                         # create

requirements.txt                       # modify (add trl, llamafactory)
```

---

## Task 1: Scaffold folder structure and update requirements

**Files:**
- Create: `04-reinforcement-learning/` (empty dirs)
- Modify: `requirements.txt`

- [ ] **Step 1: Create all directories**

```bash
mkdir -p 04-reinforcement-learning/01-fundamentals
mkdir -p 04-reinforcement-learning/02-on-policy
mkdir -p 04-reinforcement-learning/03-off-policy
mkdir -p 04-reinforcement-learning/docs
mkdir -p 04-reinforcement-learning/appendix-llamafactory/configs
```

- [ ] **Step 2: Add dependencies to requirements.txt**

Open `requirements.txt` and append:

```text
# Reinforcement learning
trl>=0.8.0
accelerate>=0.26.0
llamafactory  # appendix only
```

- [ ] **Step 3: Verify trl installs correctly**

```bash
pip install "trl>=0.8.0" accelerate --quiet
python -c "import trl; print(trl.__version__)"
```

Expected: prints a version string like `0.8.x` or higher with no errors.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt 04-reinforcement-learning/
git commit -m "feat: scaffold 04-reinforcement-learning module"
```

---

## Task 2: Create 01-fundamentals/data.jsonl

**Files:**
- Create: `04-reinforcement-learning/01-fundamentals/data.jsonl`

This file contains 25 prompt/response pairs used in the reward demo and as PPO training prompts. It also establishes the dataset domain (helpful assistant) used throughout the module.

- [ ] **Step 1: Create the data file**

Create `04-reinforcement-learning/01-fundamentals/data.jsonl` with the following content (one JSON object per line):

```jsonl
{"prompt": "What is the capital of France?", "response": "The capital of France is Paris. It is located in northern France on the Seine River and has been the country's capital since the 10th century."}
{"prompt": "Explain photosynthesis in simple terms.", "response": "Photosynthesis is how plants make food. They use sunlight, water from the soil, and carbon dioxide from the air to produce glucose and oxygen."}
{"prompt": "What is machine learning?", "response": "Machine learning is a type of artificial intelligence where computers learn patterns from data instead of being explicitly programmed. The more data the system sees, the better it gets at making predictions."}
{"prompt": "How do I stay focused while studying?", "response": "To stay focused: remove phone distractions, study in 25-minute blocks with 5-minute breaks (Pomodoro technique), stay hydrated, and review material shortly after learning it."}
{"prompt": "What causes rainbows?", "response": "Rainbows form when sunlight enters water droplets in the air. The light slows down, bends (refracts), reflects off the back of the droplet, and exits at an angle that separates it into its color spectrum."}
{"prompt": "What is the difference between RAM and storage?", "response": "RAM is temporary memory your computer uses while running programs — it clears when you power off. Storage (HDD/SSD) is permanent memory where files and programs are saved."}
{"prompt": "How does a vaccine work?", "response": "A vaccine trains your immune system to recognize a specific pathogen without causing the actual disease. It introduces a harmless version (or piece) of the pathogen so your body learns to build antibodies against it."}
{"prompt": "What is compound interest?", "response": "Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods. Over time, this causes exponential growth — often called 'interest on interest'."}
{"prompt": "Explain the water cycle.", "response": "The water cycle has four main stages: evaporation (water turns to vapor from heat), condensation (vapor forms clouds), precipitation (rain or snow falls), and collection (water gathers in oceans and rivers to restart the cycle)."}
{"prompt": "What is a neural network?", "response": "A neural network is a computing system loosely inspired by the human brain. It consists of layers of nodes (neurons) that process input data, learn patterns through training, and produce outputs like predictions or classifications."}
{"prompt": "How do I improve my writing?", "response": "Improve writing by: reading widely in your target genre, writing daily even briefly, revising ruthlessly (cut unnecessary words), getting feedback from others, and studying sentence variety and paragraph structure."}
{"prompt": "What is DNA?", "response": "DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions for the development, functioning, and reproduction of all known living organisms. It is shaped like a double helix and stored in cell nuclei."}
{"prompt": "Explain supply and demand.", "response": "Supply and demand describes how prices are set in markets. When demand for a product rises or supply falls, prices increase. When supply rises or demand falls, prices decrease. The equilibrium price is where supply equals demand."}
{"prompt": "What is the greenhouse effect?", "response": "The greenhouse effect occurs when gases like CO2 and methane trap heat from the sun in Earth's atmosphere instead of letting it escape. A natural amount is necessary for life, but excess greenhouse gases from human activity are warming the planet."}
{"prompt": "How does encryption work?", "response": "Encryption converts readable data into an unreadable format using a key. Only someone with the correct key can decrypt and read it. Modern encryption (like AES or RSA) uses complex mathematical operations that are computationally infeasible to reverse without the key."}
{"prompt": "What is the difference between supervised and unsupervised learning?", "response": "Supervised learning trains on labeled data (input-output pairs) to learn a mapping. Unsupervised learning finds structure in unlabeled data, like clustering similar items together or reducing dimensionality."}
{"prompt": "How do I manage stress effectively?", "response": "Effective stress management: exercise regularly (releases endorphins), practice mindfulness or meditation, maintain consistent sleep, identify your stressors and address root causes, and build social support networks."}
{"prompt": "What is quantum computing?", "response": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical bits (0 or 1), quantum bits (qubits) can exist in multiple states simultaneously, enabling certain computations much faster than classical computers."}
{"prompt": "Explain the difference between speed and velocity.", "response": "Speed is a scalar quantity — it measures how fast an object moves (e.g., 60 km/h). Velocity is a vector — it includes both speed and direction (e.g., 60 km/h north). An object moving in a circle has constant speed but changing velocity."}
{"prompt": "What is open source software?", "response": "Open source software has its source code publicly available for anyone to view, modify, and distribute. Examples include Linux, Python, and Firefox. It promotes collaboration, transparency, and free use, though licenses vary in restrictions."}
{"prompt": "How does the internet work?", "response": "The internet is a global network of computers. When you request a webpage, your device sends a data packet to a DNS server that translates the domain to an IP address, then routers forward your request to the destination server, which sends data back in packets."}
{"prompt": "What is inflation?", "response": "Inflation is the rate at which the general level of prices for goods and services rises over time, reducing purchasing power. It is measured by the Consumer Price Index (CPI). Moderate inflation (~2%) is considered healthy; high inflation erodes savings."}
{"prompt": "Explain how GPS works.", "response": "GPS works using a network of satellites orbiting Earth. Your device receives signals from at least four satellites and calculates its position using the time it takes each signal to arrive (trilateration). The slight time differences let it pinpoint location within meters."}
{"prompt": "What is recursion in programming?", "response": "Recursion is when a function calls itself to solve a problem by breaking it into smaller subproblems of the same type. Each recursive call works on a smaller input until it reaches a base case that stops the recursion. Classic examples: factorial, Fibonacci, tree traversal."}
{"prompt": "How do airplanes fly?", "response": "Airplanes fly using lift. Wings are shaped so air flows faster over the curved top surface than the flat bottom, creating lower pressure above the wing. This pressure difference generates an upward force (lift) that overcomes gravity when the plane reaches sufficient speed."}
```

- [ ] **Step 2: Verify the file is valid JSONL**

```bash
python -c "
import json
with open('04-reinforcement-learning/01-fundamentals/data.jsonl') as f:
    records = [json.loads(line) for line in f]
print(f'Loaded {len(records)} records')
print('Keys:', list(records[0].keys()))
"
```

Expected output:
```
Loaded 25 records
Keys: ['prompt', 'response']
```

- [ ] **Step 3: Commit**

```bash
git add 04-reinforcement-learning/01-fundamentals/data.jsonl
git commit -m "feat: add fundamentals training data (25 prompt/response pairs)"
```

---

## Task 3: Create 01-fundamentals/reward_demo.py

**Files:**
- Create: `04-reinforcement-learning/01-fundamentals/reward_demo.py`

This script scores the data.jsonl responses using three rule-based reward functions — no model training, no GPU needed. Its purpose is to show that a reward is just a scalar number.

- [ ] **Step 1: Create the script**

Create `04-reinforcement-learning/01-fundamentals/reward_demo.py`:

```python
import json
from pathlib import Path


def length_reward(response: str) -> float:
    """Reward longer, more detailed responses (normalized 0-1, peaks at 80 words)."""
    word_count = len(response.split())
    return min(word_count / 80.0, 1.0)


def keyword_reward(response: str) -> float:
    """Reward responses containing explanation markers."""
    markers = ["because", "therefore", "which means", "for example", "such as", "this is"]
    hits = sum(1 for m in markers if m.lower() in response.lower())
    return min(hits / 2.0, 1.0)


def specificity_reward(response: str) -> float:
    """Reward responses with numbers or proper nouns (indicates specificity)."""
    has_number = any(c.isdigit() for c in response)
    words = response.split()
    capitalized = sum(1 for w in words[1:] if w[0].isupper()) if len(words) > 1 else 0
    score = (0.5 if has_number else 0.0) + min(capitalized / 3.0 * 0.5, 0.5)
    return round(score, 2)


def combined_reward(response: str) -> float:
    """Weighted combination of the three reward signals."""
    return round(
        0.4 * length_reward(response)
        + 0.3 * keyword_reward(response)
        + 0.3 * specificity_reward(response),
        3,
    )


def main():
    data_path = Path(__file__).parent / "data.jsonl"
    records = [json.loads(line) for line in data_path.read_text().splitlines()]

    print("=" * 70)
    print("REWARD DEMO: Scoring responses with rule-based reward functions")
    print("=" * 70)
    print(f"\nReward functions:")
    print("  length_reward     — rewards longer, more detailed responses")
    print("  keyword_reward    — rewards explanation markers (because, e.g.)")
    print("  specificity_reward — rewards numbers and proper nouns")
    print("  combined_reward   — weighted average of all three\n")

    print(f"{'Prompt':<45} {'Len':>5} {'Kw':>5} {'Spec':>5} {'Total':>7}")
    print("-" * 70)

    scores = []
    for record in records:
        prompt = record["prompt"][:43] + ".." if len(record["prompt"]) > 45 else record["prompt"]
        r = record["response"]
        l_r = length_reward(r)
        k_r = keyword_reward(r)
        s_r = specificity_reward(r)
        c_r = combined_reward(r)
        scores.append(c_r)
        print(f"{prompt:<45} {l_r:>5.2f} {k_r:>5.2f} {s_r:>5.2f} {c_r:>7.3f}")

    print("-" * 70)
    print(f"\nAverage combined reward: {sum(scores)/len(scores):.3f}")
    print(f"Min: {min(scores):.3f}  Max: {max(scores):.3f}")
    print("\nKey insight: a reward is just a scalar number per response.")
    print("RL training optimizes the model to produce higher-scoring responses.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and verify output**

```bash
cd 04-reinforcement-learning/01-fundamentals
python reward_demo.py
```

Expected: a table of 25 rows with reward scores, no errors, and a summary line at the bottom.

- [ ] **Step 3: Commit**

```bash
git add 04-reinforcement-learning/01-fundamentals/reward_demo.py
git commit -m "feat: add reward_demo.py (rule-based reward scoring, no training needed)"
```

---

## Task 4: Create 01-fundamentals/README.md

**Files:**
- Create: `04-reinforcement-learning/01-fundamentals/README.md`

- [ ] **Step 1: Create the README**

Create `04-reinforcement-learning/01-fundamentals/README.md`:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add 04-reinforcement-learning/01-fundamentals/README.md
git commit -m "feat: add 01-fundamentals README"
```

---

## Task 5: Create 02-on-policy/reward_model.py

**Files:**
- Create: `04-reinforcement-learning/02-on-policy/reward_model.py`

This module defines a `SentimentRewardModel` — a thin `nn.Module` wrapper around DistilBERT fine-tuned on SST-2 (positive/negative sentiment). It uses positive sentiment probability as the reward signal. Teaching focus: the reward model is just a scorer that returns a scalar per response.

- [ ] **Step 1: Create the module**

Create `04-reinforcement-learning/02-on-policy/reward_model.py`:

```python
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REWARD_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


class SentimentRewardModel(nn.Module):
    """
    Reward model that scores LLM responses by positive sentiment probability.

    Returns a scalar reward in [0, 1] for each response:
      - 1.0 = very positive / confident tone
      - 0.0 = very negative / uncertain tone

    Uses DistilBERT SST-2 (66M params) — fast enough on CPU/MPS.
    """

    def __init__(self):
        super().__init__()
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL_NAME
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        with torch.no_grad():
            logits = self.classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        # SST-2 label 1 = POSITIVE, return its probability as reward
        return logits.softmax(dim=-1)[:, 1]


def load_reward_model(device: str) -> SentimentRewardModel:
    model = SentimentRewardModel().to(device)
    model.eval()
    return model


def load_reward_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)


if __name__ == "__main__":
    # Quick sanity check: score two responses
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    model = load_reward_model(device)
    tokenizer = load_reward_tokenizer()

    samples = [
        "Paris is the beautiful capital of France, known for art and culture.",
        "I don't know. It might be Paris, or maybe not. Hard to say.",
    ]

    inputs = tokenizer(samples, return_tensors="pt", padding=True, truncation=True).to(device)
    rewards = model(**inputs)

    for text, reward in zip(samples, rewards):
        print(f"Reward: {reward:.3f} | {text[:60]}")
```

- [ ] **Step 2: Run the sanity check**

```bash
cd 04-reinforcement-learning/02-on-policy
python reward_model.py
```

Expected: two lines with reward scores. The first (confident answer) should score higher than the second (uncertain answer). No errors.

- [ ] **Step 3: Commit**

```bash
git add 04-reinforcement-learning/02-on-policy/reward_model.py
git commit -m "feat: add SentimentRewardModel (DistilBERT SST-2 wrapper)"
```

---

## Task 6: Create 02-on-policy/ppo_train.py

**Files:**
- Create: `04-reinforcement-learning/02-on-policy/ppo_train.py`

PPO training using `trl.experimental.ppo`. Note: PPO requires four model instances (policy, reference, value, reward). This is resource-intensive — the script detects device and adjusts accordingly. For MPS/CPU, use `per_device_train_batch_size=2`.

- [ ] **Step 1: Create the training script**

Create `04-reinforcement-learning/02-on-policy/ppo_train.py`:

```python
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
```

- [ ] **Step 2: Verify the script is importable (dry run)**

```bash
cd 04-reinforcement-learning/02-on-policy
python -c "
import ast, sys
with open('ppo_train.py') as f:
    source = f.read()
ast.parse(source)
print('ppo_train.py: syntax OK')
"
```

Expected: `ppo_train.py: syntax OK`

- [ ] **Step 3: Commit**

```bash
git add 04-reinforcement-learning/02-on-policy/ppo_train.py
git commit -m "feat: add ppo_train.py (on-policy PPO via trl.experimental.ppo)"
```

---

## Task 7: Create 02-on-policy/inference.py

**Files:**
- Create: `04-reinforcement-learning/02-on-policy/inference.py`

Loads both the base model and PPO-trained checkpoint, runs the same prompts through both, and prints a side-by-side comparison so learners can see what PPO changed.

- [ ] **Step 1: Create the inference script**

Create `04-reinforcement-learning/02-on-policy/inference.py`:

```python
"""
Compare base model vs PPO-trained model on the same prompts.
Run this after ppo_train.py has completed.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PPO_MODEL = "outputs/ppo/final"

PROMPTS = [
    "What is the capital of France?",
    "How does a vaccine work?",
    "Explain compound interest.",
]

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 120) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    print("Loading base model...")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)

    print("Loading PPO-trained model...")
    ppo_tokenizer = AutoTokenizer.from_pretrained(PPO_MODEL)
    ppo_model = AutoModelForCausalLM.from_pretrained(PPO_MODEL).to(device)

    for prompt in PROMPTS:
        print("\n" + "=" * 70)
        print(f"PROMPT: {prompt}")
        print("-" * 70)
        print(f"BASE MODEL:\n{generate(base_model, base_tokenizer, prompt)}")
        print("-" * 70)
        print(f"PPO MODEL:\n{generate(ppo_model, ppo_tokenizer, prompt)}")

    print("\n" + "=" * 70)
    print("Compare: does the PPO model give more confident, positive responses?")
    print("That reflects the sentiment reward signal it was trained on.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "
import ast
with open('04-reinforcement-learning/02-on-policy/inference.py') as f:
    ast.parse(f.read())
print('inference.py: syntax OK')
"
```

Expected: `inference.py: syntax OK`

- [ ] **Step 3: Commit**

```bash
git add 04-reinforcement-learning/02-on-policy/inference.py
git commit -m "feat: add ppo inference.py (base vs PPO-trained comparison)"
```

---

## Task 8: Create 02-on-policy/README.md

**Files:**
- Create: `04-reinforcement-learning/02-on-policy/README.md`

- [ ] **Step 1: Create the README**

Create `04-reinforcement-learning/02-on-policy/README.md`:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add 04-reinforcement-learning/02-on-policy/README.md
git commit -m "feat: add 02-on-policy README"
```

---

## Task 9: Create 03-off-policy/preference_data.jsonl

**Files:**
- Create: `04-reinforcement-learning/03-off-policy/preference_data.jsonl`

DPO requires `(prompt, chosen, rejected)` triplets. Each pair shows the model which response is preferred. This dataset covers the same domain as `data.jsonl` but deliberately contrasts helpful vs unhelpful responses.

- [ ] **Step 1: Create the preference data file**

Create `04-reinforcement-learning/03-off-policy/preference_data.jsonl`:

```jsonl
{"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris. It has been the country's political and cultural center since the 10th century and is home to landmarks like the Eiffel Tower and the Louvre.", "rejected": "I think it might be Paris? Or maybe Lyon. I'm not entirely sure, you should probably look it up."}
{"prompt": "Explain photosynthesis in simple terms.", "chosen": "Photosynthesis is how plants make food. They absorb sunlight, take in CO2 from the air and water from the soil, and convert these into glucose (energy) and oxygen. The oxygen is released into the air.", "rejected": "Photosynthesis is a complex biochemical process involving chlorophyll and the Calvin cycle. It's quite complicated to explain simply."}
{"prompt": "What is machine learning?", "chosen": "Machine learning is a branch of AI where computers learn patterns from data rather than being explicitly programmed. A spam filter, for example, learns to recognize spam by studying thousands of emails — no one writes rules for every case.", "rejected": "Machine learning is when machines learn. It involves algorithms and data and is used in many applications today."}
{"prompt": "How do I stay focused while studying?", "chosen": "Try the Pomodoro technique: study for 25 minutes, break for 5. Remove your phone from the room, use noise-canceling headphones, and set a specific goal for each session (e.g., 'finish chapter 3'). Review notes within 24 hours to retain them.", "rejected": "Just try to focus more. Maybe drink some coffee. Everyone struggles with this."}
{"prompt": "What causes rainbows?", "chosen": "Rainbows form when sunlight enters water droplets in the air. The light bends (refracts), reflects off the inside of the droplet, then bends again as it exits. Different wavelengths bend at different angles, separating white light into the visible spectrum.", "rejected": "Rainbows are caused by light and water. They appear after it rains when the sun is out."}
{"prompt": "What is the difference between RAM and storage?", "chosen": "RAM is your computer's short-term memory — fast but temporary, cleared when you power off. Storage (SSD or HDD) is long-term memory where your files and programs live permanently. When you open a file, it moves from storage into RAM to be processed.", "rejected": "RAM and storage are both types of memory. RAM is random access memory. Storage is for storing things. They are different in various ways."}
{"prompt": "How does a vaccine work?", "chosen": "A vaccine introduces a harmless version or fragment of a pathogen (like a protein from its outer coat). Your immune system learns to recognize it and builds antibodies. If you later encounter the real pathogen, your immune system recognizes it immediately and responds faster.", "rejected": "Vaccines work by helping your immune system. They contain stuff that makes you immune to diseases. Doctors recommend them."}
{"prompt": "What is compound interest?", "chosen": "Compound interest earns interest on both your principal and previously earned interest. Example: $1,000 at 10% annual return becomes $1,100 after year 1, $1,210 after year 2 (interest on $1,100), $1,331 after year 3. Over decades, this creates exponential growth.", "rejected": "Compound interest means you get interest over time. It's better than simple interest. Banks use it for savings accounts."}
{"prompt": "Explain the water cycle.", "chosen": "The water cycle moves water through four stages: (1) Evaporation — heat turns surface water into vapor, (2) Condensation — vapor cools and forms clouds, (3) Precipitation — water falls as rain or snow, (4) Collection — water gathers in rivers and oceans, restarting the cycle.", "rejected": "Water goes up as vapor and comes back down as rain. This repeats constantly. It's called the water cycle or hydrological cycle."}
{"prompt": "What is a neural network?", "chosen": "A neural network processes data through layers of interconnected nodes. Each node applies a weighted transformation and passes the result forward. During training, the weights adjust to minimize prediction error via backpropagation. Deep networks (many layers) can learn complex patterns like image recognition.", "rejected": "Neural networks are inspired by the brain. They have neurons and connections. They are used in deep learning and AI research."}
{"prompt": "How do I improve my writing?", "chosen": "Three high-leverage habits: (1) Read deliberately — analyze why a piece works, not just what it says. (2) Write daily, even 150 words. (3) Cut ruthlessly — every sentence must earn its place. Also: read your work aloud to catch awkward phrasing.", "rejected": "Practice makes perfect. Read a lot and write a lot. Get feedback from people you trust. It takes time to improve."}
{"prompt": "What is DNA?", "chosen": "DNA is the double-helix molecule that encodes genetic instructions. It is made of four nucleotide bases (A, T, C, G) — the sequence of these bases spells out genes. Genes are instructions for making proteins, which carry out virtually all functions in living cells.", "rejected": "DNA stands for deoxyribonucleic acid. It carries genetic information and is found in cells. It determines your traits and characteristics."}
{"prompt": "Explain supply and demand.", "chosen": "Supply and demand determines prices through competition. When more people want a product than is available (high demand, low supply), sellers can charge more. When supply exceeds demand, prices fall. The price where they balance is called equilibrium. This mechanism drives most market prices.", "rejected": "Supply and demand is an economic concept. When supply goes up, prices go down. When demand goes up, prices go up. It's a fundamental principle in economics."}
{"prompt": "What is the greenhouse effect?", "chosen": "The greenhouse effect works like a car on a sunny day: the sun's energy passes through the atmosphere and warms the Earth, but greenhouse gases (CO2, methane) trap some of the heat trying to escape. A natural greenhouse effect is essential for life — the problem is human activity intensifying it.", "rejected": "The greenhouse effect is when gases trap heat. CO2 is the main culprit. It causes climate change and global warming which is a serious issue today."}
{"prompt": "How does encryption work?", "chosen": "Encryption scrambles data using a mathematical key. AES (symmetric) uses one key to lock and unlock data — fast, used for storage. RSA (asymmetric) uses a public key to encrypt and a private key to decrypt — used for secure key exchange. Without the key, brute-forcing modern encryption would take longer than the universe's age.", "rejected": "Encryption makes data secure by scrambling it. You need a key to read it. It's used in HTTPS and messaging apps to protect privacy."}
{"prompt": "What is the difference between supervised and unsupervised learning?", "chosen": "Supervised learning uses labeled data (input + correct output) to train a model — like teaching with an answer key. Unsupervised learning finds structure in unlabeled data — like sorting a pile of objects by similarity without being told the categories. Clustering and dimensionality reduction are common unsupervised techniques.", "rejected": "Supervised learning has labels. Unsupervised doesn't have labels. Both are types of machine learning used for different purposes in AI."}
{"prompt": "How do I manage stress effectively?", "chosen": "The most evidence-backed approaches: (1) Regular aerobic exercise (30 min, 3x/week) reduces cortisol. (2) Sleep 7-9 hours — sleep deprivation amplifies stress. (3) Identify your stressor specifically and write down one action you can take. Vague worrying is more exhausting than problem-solving.", "rejected": "Try to relax and take breaks. Meditation can help. Talk to friends or family. Don't overthink things. Everyone gets stressed sometimes."}
{"prompt": "What is quantum computing?", "chosen": "Classical computers use bits (0 or 1). Quantum computers use qubits, which can be 0, 1, or a superposition of both simultaneously. This allows quantum computers to explore many solutions at once. For specific problems like factoring large numbers or simulating molecules, quantum computers are exponentially faster than classical ones.", "rejected": "Quantum computing uses quantum mechanics. It's very different from normal computers and will be very powerful in the future. Companies like IBM and Google are working on it."}
{"prompt": "Explain the difference between speed and velocity.", "chosen": "Speed is a scalar — it only has magnitude (60 km/h). Velocity is a vector — it has magnitude and direction (60 km/h north). This distinction matters: a car driving in a circle at constant speed has constantly changing velocity (direction changes), meaning it is always accelerating.", "rejected": "Speed tells you how fast something is going. Velocity also includes direction. They are related but different concepts in physics."}
{"prompt": "What is open source software?", "chosen": "Open source software publishes its source code under a license that allows anyone to view, modify, and distribute it. The GPL (copyleft) requires derivatives to stay open source; MIT and Apache licenses are more permissive. Linux, Python, and VS Code are major examples. The model enables peer review, community contribution, and free use.", "rejected": "Open source software is free and anyone can see the code. It's the opposite of proprietary software. Many popular programs are open source."}
{"prompt": "How does the internet work?", "chosen": "When you visit a website: (1) Your browser asks a DNS server to translate the domain to an IP address. (2) Your request travels through routers (BGP protocol) to the destination server. (3) The server responds with HTML/CSS/JS packets. (4) TCP ensures all packets arrive and reassembles them. This round trip often completes in under 100ms.", "rejected": "The internet connects computers around the world. Data travels through cables and wireless signals. Websites are hosted on servers that send data to your device."}
{"prompt": "What is inflation?", "chosen": "Inflation is the rate at which prices rise, reducing purchasing power. It is measured by the Consumer Price Index (CPI). Causes include increased money supply (too much money chasing goods), supply chain disruptions, or strong demand. Central banks target ~2% annual inflation as a sign of a healthy, growing economy.", "rejected": "Inflation means prices go up. It's bad for consumers because things cost more. The government and central banks try to control it."}
{"prompt": "Explain how GPS works.", "chosen": "Your GPS receiver listens to signals from at least 4 satellites. Each signal carries a timestamp. Your device calculates how long each signal took to arrive (speed of light × time = distance). With 4 distances, it can pinpoint your 3D position via trilateration. Accuracy is typically 3-5 meters for consumer devices.", "rejected": "GPS uses satellites to find your location. Your phone receives signals from satellites and calculates where you are. It's very accurate and useful for navigation."}
{"prompt": "What is recursion in programming?", "chosen": "Recursion is a function that calls itself with a smaller input until a base case stops it. Example: factorial(5) = 5 × factorial(4) = 5 × 4 × factorial(3)... until factorial(0) = 1. Every recursive solution has two parts: the base case (stop condition) and the recursive case (reduce the problem). Recursion is elegant for tree traversal and divide-and-conquer algorithms.", "rejected": "Recursion is when a function calls itself. It's used in programming for certain types of problems. You need a base case or it will loop forever."}
{"prompt": "How do airplanes fly?", "chosen": "Wings generate lift through two mechanisms: (1) Bernoulli's principle — the curved upper surface makes air travel faster, lowering pressure above the wing. (2) Newton's third law — the wing deflects air downward; air pushes the wing upward. Both effects contribute. Lift overcomes gravity when the plane reaches takeoff speed (~270 km/h for a 737).", "rejected": "Airplanes fly because of their wings. Air flows differently over the curved wing shape creating lift. Engines provide thrust to move forward."}
```

- [ ] **Step 2: Verify the file**

```bash
python -c "
import json
with open('04-reinforcement-learning/03-off-policy/preference_data.jsonl') as f:
    records = [json.loads(line) for line in f]
print(f'Records: {len(records)}')
print(f'Keys: {list(records[0].keys())}')
# Check all records have correct keys
assert all('prompt' in r and 'chosen' in r and 'rejected' in r for r in records)
print('All records valid.')
"
```

Expected:
```
Records: 25
Keys: ['prompt', 'chosen', 'rejected']
All records valid.
```

- [ ] **Step 3: Commit**

```bash
git add 04-reinforcement-learning/03-off-policy/preference_data.jsonl
git commit -m "feat: add preference_data.jsonl (25 prompt/chosen/rejected triplets)"
```

---

## Task 10: Create 03-off-policy/dpo_train.py

**Files:**
- Create: `04-reinforcement-learning/03-off-policy/dpo_train.py`

DPO training using `trl.DPOTrainer`. No reward model needed — preference pairs are the only training signal.

- [ ] **Step 1: Create the training script**

Create `04-reinforcement-learning/03-off-policy/dpo_train.py`:

```python
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
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "
import ast
with open('04-reinforcement-learning/03-off-policy/dpo_train.py') as f:
    ast.parse(f.read())
print('dpo_train.py: syntax OK')
"
```

Expected: `dpo_train.py: syntax OK`

- [ ] **Step 3: Commit**

```bash
git add 04-reinforcement-learning/03-off-policy/dpo_train.py
git commit -m "feat: add dpo_train.py (off-policy DPO via trl.DPOTrainer)"
```

---

## Task 11: Create 03-off-policy/grpo_train.py

**Files:**
- Create: `04-reinforcement-learning/03-off-policy/grpo_train.py`

GRPO training using `trl.GRPOTrainer`. Uses a custom reward function instead of a separate reward model. The script generates groups of responses per prompt and scores them relatively.

- [ ] **Step 1: Create the training script**

Create `04-reinforcement-learning/03-off-policy/grpo_train.py`:

```python
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
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "
import ast
with open('04-reinforcement-learning/03-off-policy/grpo_train.py') as f:
    ast.parse(f.read())
print('grpo_train.py: syntax OK')
"
```

Expected: `grpo_train.py: syntax OK`

- [ ] **Step 3: Commit**

```bash
git add 04-reinforcement-learning/03-off-policy/grpo_train.py
git commit -m "feat: add grpo_train.py (GRPO via trl.GRPOTrainer, custom reward fn)"
```

---

## Task 12: Create 03-off-policy/inference.py

**Files:**
- Create: `04-reinforcement-learning/03-off-policy/inference.py`

- [ ] **Step 1: Create the script**

Create `04-reinforcement-learning/03-off-policy/inference.py`:

```python
"""
Compare DPO-trained vs GRPO-trained model on the same prompts.
Run after dpo_train.py and grpo_train.py have completed.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DPO_MODEL = "outputs/dpo/final"
GRPO_MODEL = "outputs/grpo/final"

PROMPTS = [
    "What is compound interest?",
    "How does encryption work?",
    "What is the difference between supervised and unsupervised learning?",
]

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


def load(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    return mdl, tok


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    missing = [p for p in [DPO_MODEL, GRPO_MODEL] if not Path(p).exists()]
    if missing:
        print(f"Missing checkpoints: {missing}")
        print("Run dpo_train.py and grpo_train.py first.")
        return

    print("Loading DPO model...")
    dpo_model, dpo_tok = load(DPO_MODEL)
    print("Loading GRPO model...")
    grpo_model, grpo_tok = load(GRPO_MODEL)

    for prompt in PROMPTS:
        print("\n" + "=" * 70)
        print(f"PROMPT: {prompt}")
        print("-" * 70)
        print(f"DPO (preference pairs):\n{generate(dpo_model, dpo_tok, prompt)}")
        print("-" * 70)
        print(f"GRPO (group relative):\n{generate(grpo_model, grpo_tok, prompt)}")

    print("\n" + "=" * 70)
    print("Observation guide:")
    print("  DPO: shaped by human preference pairs — tends toward 'chosen' style")
    print("  GRPO: shaped by reward function — tends toward length/structure targets")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "
import ast
with open('04-reinforcement-learning/03-off-policy/inference.py') as f:
    ast.parse(f.read())
print('inference.py: syntax OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add 04-reinforcement-learning/03-off-policy/inference.py
git commit -m "feat: add off-policy inference.py (DPO vs GRPO comparison)"
```

---

## Task 13: Create 03-off-policy/README.md

**Files:**
- Create: `04-reinforcement-learning/03-off-policy/README.md`

- [ ] **Step 1: Create the README**

Create `04-reinforcement-learning/03-off-policy/README.md`:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add 04-reinforcement-learning/03-off-policy/README.md
git commit -m "feat: add 03-off-policy README"
```

---

## Task 14: Create docs/ (three concept documents)

**Files:**
- Create: `04-reinforcement-learning/docs/01-rl-fundamentals.md`
- Create: `04-reinforcement-learning/docs/02-on-vs-off-policy.md`
- Create: `04-reinforcement-learning/docs/03-algorithm-comparison.md`

- [ ] **Step 1: Create 01-rl-fundamentals.md**

Create `04-reinforcement-learning/docs/01-rl-fundamentals.md`:

```markdown
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
```

- [ ] **Step 2: Create 02-on-vs-off-policy.md**

Create `04-reinforcement-learning/docs/02-on-vs-off-policy.md`:

```markdown
# On-Policy vs Off-Policy

The central distinction in this module.

## On-Policy (PPO)

> The data used to update the policy was generated by the **current** policy.

After each gradient update, the old data is stale — the model must generate fresh responses with its new weights. This is why PPO alternates between a **generation phase** and an **update phase**.

```
Generate with policy_v1 → update to policy_v2 → discard policy_v1 data
Generate with policy_v2 → update to policy_v3 → discard policy_v2 data
...
```

**Trade-offs:**
- More stable training (data always reflects current behavior)
- Sample inefficient (old data is discarded)
- Requires live reward model during training
- Needs value network (adds ~30% memory)

## Off-Policy (DPO / GRPO)

> The training data was collected independently of the current training run.

DPO trains on human-labeled preference pairs collected before training starts. The data doesn't change during training.

GRPO generates data from the current model (so technically on-policy generation), but uses group-relative normalization to avoid the value network, and is architecturally closer to DPO in infrastructure requirements.

**Trade-offs:**
- More sample efficient (data can be reused)
- No live reward model needed (DPO) or simpler reward function (GRPO)
- Simpler infrastructure (2 models instead of 4)
- Data quality depends entirely on preference data or reward function design

## Decision Table

| Question | On-Policy (PPO) | Off-Policy (DPO/GRPO) |
|---|---|---|
| Do you have preference data? | Not required | Required (DPO) or helpful |
| Can you build a reward model? | Required | Not required |
| Memory budget? | 4× model size | 2× model size |
| Training stability? | More stable | Can be noisier |
| Production use today? | Less common (complex) | More common (simpler) |

## In Practice (2024–2025)

Most production LLM alignment pipelines use DPO or GRPO rather than PPO because:
1. They are simpler to implement (no value network, no live reward model)
2. They often achieve comparable or better results
3. GRPO scales well and was central to DeepSeek-R1's reasoning improvements

PPO remains important for research and for tasks requiring precise, step-level credit assignment.
```

- [ ] **Step 3: Create 03-algorithm-comparison.md**

Create `04-reinforcement-learning/docs/03-algorithm-comparison.md`:

```markdown
# Algorithm Comparison: PPO vs DPO vs GRPO

## Summary Table

| | PPO | DPO | GRPO |
|---|---|---|---|
| **Family** | On-policy | Off-policy | On-policy generation, off-policy style |
| **Training data** | Prompts | Preference pairs | Prompts |
| **Reward signal** | Reward model (live) | Implicit from pairs | Reward function (callable) |
| **Value network** | Required | No | No |
| **Memory (vs base)** | ~4× | ~2× | ~2× |
| **Stability** | High | Medium | Medium |
| **Complexity** | High | Low | Medium |
| **TRL class** | `trl.experimental.ppo.PPOTrainer` | `trl.DPOTrainer` | `trl.GRPOTrainer` |
| **LlamaFactory stage** | `ppo` | `dpo` | `grpo` |

## When to Choose Each

### PPO
- You need fine-grained, token-level credit assignment
- You have a reliable reward model (e.g., a strong helpfulness classifier)
- Stability is critical (e.g., production RLHF pipeline at scale)
- You have sufficient GPU memory

### DPO
- You have high-quality human preference data or a strong judge model
- You want the simplest implementation with fewest moving parts
- You are aligning to a style or set of values (not optimizing a metric)
- You want fast iteration

### GRPO
- You have a clear, measurable objective (math correctness, code tests, format)
- You want automated reward without human labeling
- You need PPO-like RL signal but with less memory (no value network)
- Your task benefits from comparative reasoning across multiple attempts

## The Modern Consensus (2025)

- **Research:** PPO for precise alignment studies; GRPO for reasoning tasks
- **Production alignment:** DPO dominates due to simplicity and effectiveness
- **Reasoning models:** GRPO (DeepSeek-R1, Qwen-thinking) — reward on final answer correctness drives chain-of-thought emergence
- **Safety fine-tuning:** DPO with constitutional AI preference data

## Reference Papers

- PPO: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- RLHF with PPO: [Ouyang et al., 2022 (InstructGPT)](https://arxiv.org/abs/2203.02155)
- DPO: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)
- GRPO: [Shao et al., 2024 (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
- DeepSeek-R1: [DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)
```

- [ ] **Step 4: Commit all docs**

```bash
git add 04-reinforcement-learning/docs/
git commit -m "feat: add RL concept docs (fundamentals, on-vs-off-policy, algorithm comparison)"
```

---

## Task 15: Create appendix-llamafactory/ configs and run.sh

**Files:**
- Create: `04-reinforcement-learning/appendix-llamafactory/configs/ppo_qwen.yaml`
- Create: `04-reinforcement-learning/appendix-llamafactory/configs/dpo_qwen.yaml`
- Create: `04-reinforcement-learning/appendix-llamafactory/configs/grpo_qwen.yaml`
- Create: `04-reinforcement-learning/appendix-llamafactory/run.sh`

LlamaFactory YAML format: `stage` sets the training method; `finetuning_type: lora` keeps it memory-efficient. Official docs: https://llamafactory.readthedocs.io/en/latest/

- [ ] **Step 1: Create ppo_qwen.yaml**

Create `04-reinforcement-learning/appendix-llamafactory/configs/ppo_qwen.yaml`:

```yaml
### PPO Training with LlamaFactory
### Equivalent to: 02-on-policy/ppo_train.py
### Docs: https://llamafactory.readthedocs.io/en/latest/advanced/trainers.html

### Model
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct

### Method
stage: ppo
do_train: true
finetuning_type: lora
lora_target: all

# Path to a trained reward model (required for PPO)
# Train a reward model first with: stage: rm
reward_model: outputs/reward_model
reward_model_type: lora

### Dataset
# Use built-in demo dataset or replace with your own
dataset: alpaca_en_demo
template: qwen
cutoff_len: 1024

### Output
output_dir: outputs/ppo_llamafactory
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

### Training hyperparameters
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true

### PPO-specific
ppo_score_norm: true
ppo_whiten_rewards: false
```

- [ ] **Step 2: Create dpo_qwen.yaml**

Create `04-reinforcement-learning/appendix-llamafactory/configs/dpo_qwen.yaml`:

```yaml
### DPO Training with LlamaFactory
### Equivalent to: 03-off-policy/dpo_train.py
### Docs: https://llamafactory.readthedocs.io/en/latest/advanced/trainers.html

### Model
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct

### Method
stage: dpo
do_train: true
finetuning_type: lora
lora_target: all

# DPO-specific: beta controls KL penalty (higher = stay closer to reference)
pref_beta: 0.1

# Loss type: sigmoid (DPO), orpo, simpo, ipo
pref_loss: sigmoid

### Dataset
# Dataset must be in Preference format: prompt, chosen, rejected
# Use built-in demo or replace with your own preference dataset
dataset: dpo_en_demo
template: qwen
cutoff_len: 1024

### Output
output_dir: outputs/dpo_llamafactory
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

### Training hyperparameters
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
```

- [ ] **Step 3: Create grpo_qwen.yaml**

Create `04-reinforcement-learning/appendix-llamafactory/configs/grpo_qwen.yaml`:

```yaml
### GRPO Training with LlamaFactory (via EasyR1)
### Equivalent to: 03-off-policy/grpo_train.py
### Docs: https://llamafactory.readthedocs.io/en/latest/advanced/trainers.html
### EasyR1: https://github.com/hiyouga/LLaMA-Factory/tree/main/examples/train_grpo

### Model
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct

### Method
stage: grpo
do_train: true
finetuning_type: lora
lora_target: all

### Dataset
# GRPO requires a dataset with 'prompt' column and verifiable answers
# Use a math/reasoning dataset, or provide your own with reward_model
dataset: math_en_demo
template: qwen
cutoff_len: 1024

### Output
output_dir: outputs/grpo_llamafactory
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

### Training hyperparameters
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true

### GRPO-specific
# Number of responses generated per prompt (G in the paper)
grpo_num_generations: 4
```

- [ ] **Step 4: Create run.sh**

Create `04-reinforcement-learning/appendix-llamafactory/run.sh`:

```bash
#!/usr/bin/env bash
# LlamaFactory RL training — one command per method
# Each command is equivalent to the corresponding Python script in 02-on-policy/ or 03-off-policy/
#
# Prerequisites:
#   pip install llamafactory
#   llamafactory-cli train --help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== LlamaFactory RL Training ==="
echo "Choose a method to train:"
echo "  1) PPO  (on-policy)  — requires a pre-trained reward model"
echo "  2) DPO  (off-policy) — requires preference pairs dataset"
echo "  3) GRPO (off-policy) — requires prompts + verifiable reward"
echo ""

case "${1:-}" in
  ppo)
    echo "Running PPO training..."
    llamafactory-cli train "$SCRIPT_DIR/configs/ppo_qwen.yaml"
    ;;
  dpo)
    echo "Running DPO training..."
    llamafactory-cli train "$SCRIPT_DIR/configs/dpo_qwen.yaml"
    ;;
  grpo)
    echo "Running GRPO training..."
    llamafactory-cli train "$SCRIPT_DIR/configs/grpo_qwen.yaml"
    ;;
  all)
    echo "Running DPO then GRPO (skipping PPO — needs reward model first)..."
    llamafactory-cli train "$SCRIPT_DIR/configs/dpo_qwen.yaml"
    llamafactory-cli train "$SCRIPT_DIR/configs/grpo_qwen.yaml"
    ;;
  *)
    echo "Usage: ./run.sh [ppo|dpo|grpo|all]"
    echo ""
    echo "Or run directly:"
    echo "  llamafactory-cli train configs/ppo_qwen.yaml"
    echo "  llamafactory-cli train configs/dpo_qwen.yaml"
    echo "  llamafactory-cli train configs/grpo_qwen.yaml"
    exit 1
    ;;
esac

echo "Done. Check outputs/ for saved checkpoints."
```

- [ ] **Step 5: Make run.sh executable and commit**

```bash
chmod +x 04-reinforcement-learning/appendix-llamafactory/run.sh
git add 04-reinforcement-learning/appendix-llamafactory/configs/
git add 04-reinforcement-learning/appendix-llamafactory/run.sh
git commit -m "feat: add LlamaFactory YAML configs and run.sh (PPO, DPO, GRPO)"
```

---

## Task 16: Create appendix-llamafactory/README.md

**Files:**
- Create: `04-reinforcement-learning/appendix-llamafactory/README.md`

- [ ] **Step 1: Create the README**

Create `04-reinforcement-learning/appendix-llamafactory/README.md`:

```markdown
# Appendix — LlamaFactory

> The same PPO, DPO, and GRPO training from 02 and 03, with zero Python code.

[LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) is a production-grade LLM fine-tuning framework that wraps TRL, DeepSpeed, and other backends behind a unified YAML + CLI interface. It supports 100+ model architectures and all major training methods.

## Install

```bash
pip install llamafactory
```

## Run

```bash
# DPO (simplest — no reward model needed)
llamafactory-cli train configs/dpo_qwen.yaml

# GRPO (no reward model, but needs verifiable dataset)
llamafactory-cli train configs/grpo_qwen.yaml

# PPO (requires a pre-trained reward model — see note below)
llamafactory-cli train configs/ppo_qwen.yaml

# Or use the helper script
./run.sh dpo
./run.sh grpo
```

## What the YAML replaces

Each config file is equivalent to the corresponding Python script:

| YAML config | Equivalent Python |
|---|---|
| `configs/dpo_qwen.yaml` | `../03-off-policy/dpo_train.py` |
| `configs/grpo_qwen.yaml` | `../03-off-policy/grpo_train.py` |
| `configs/ppo_qwen.yaml` | `../02-on-policy/ppo_train.py` |

## Key YAML parameters

| Parameter | Purpose |
|---|---|
| `stage` | Training method: `ppo`, `dpo`, `grpo`, `sft`, `rm` |
| `finetuning_type` | `lora` (efficient) or `full` (all parameters) |
| `pref_beta` | DPO KL penalty coefficient (default: 0.1) |
| `pref_loss` | DPO loss type: `sigmoid`, `orpo`, `simpo` |
| `reward_model` | PPO only: path to trained reward model |
| `grpo_num_generations` | GRPO: responses per prompt (G in the paper) |
| `dataset` | Dataset name (built-in) or path to custom dataset |
| `template` | Chat template: `qwen`, `llama3`, `mistral`, etc. |

## Raw TRL vs LlamaFactory

| | Raw TRL (02 / 03) | LlamaFactory (this appendix) |
|---|---|---|
| **Code required** | ~100 lines Python | 0 lines (YAML only) |
| **Customization** | Full — any Python logic | Config-bound — YAML parameters only |
| **Custom reward fn** | Any callable | Must use built-in or registered functions |
| **Debugging** | Direct access to all objects | Black box — harder to inspect internals |
| **Best for** | Research, custom reward models | Standard alignment tasks, production |

**Rule of thumb:** start with raw TRL to understand the mechanics. Move to LlamaFactory when you want a reproducible, production-ready training pipeline.

## PPO note

PPO requires a pre-trained reward model. Train one first with:

```bash
# 1. Train a reward model (stage: rm)
# Create rm_qwen.yaml with stage: rm and your preference dataset
llamafactory-cli train configs/rm_qwen.yaml

# 2. Then run PPO pointing to it
llamafactory-cli train configs/ppo_qwen.yaml  # reward_model: points to step 1 output
```

## Dataset formats

LlamaFactory uses named built-in datasets or custom datasets registered in `data/dataset_info.json`.

| Training method | Required format |
|---|---|
| SFT / PPO | `{"instruction": "...", "output": "..."}` |
| DPO | `{"instruction": "...", "chosen": "...", "rejected": "..."}` |
| GRPO | `{"instruction": "...", "answer": "..."}` (verifiable answer) |

See [LlamaFactory dataset docs](https://llamafactory.readthedocs.io/en/latest/) for full format reference.
```

- [ ] **Step 2: Commit**

```bash
git add 04-reinforcement-learning/appendix-llamafactory/README.md
git commit -m "feat: add LlamaFactory appendix README"
```

---

## Task 17: Create top-level 04-reinforcement-learning/README.md

**Files:**
- Create: `04-reinforcement-learning/README.md`

- [ ] **Step 1: Create the README**

Create `04-reinforcement-learning/README.md`:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add 04-reinforcement-learning/README.md
git commit -m "feat: add top-level RL module README"
```

---

## Task 18: Final verification

- [ ] **Step 1: Verify folder structure**

```bash
find 04-reinforcement-learning -type f | sort
```

Expected output:
```
04-reinforcement-learning/README.md
04-reinforcement-learning/01-fundamentals/README.md
04-reinforcement-learning/01-fundamentals/data.jsonl
04-reinforcement-learning/01-fundamentals/reward_demo.py
04-reinforcement-learning/02-on-policy/README.md
04-reinforcement-learning/02-on-policy/inference.py
04-reinforcement-learning/02-on-policy/ppo_train.py
04-reinforcement-learning/02-on-policy/reward_model.py
04-reinforcement-learning/03-off-policy/README.md
04-reinforcement-learning/03-off-policy/dpo_train.py
04-reinforcement-learning/03-off-policy/grpo_train.py
04-reinforcement-learning/03-off-policy/inference.py
04-reinforcement-learning/03-off-policy/preference_data.jsonl
04-reinforcement-learning/docs/01-rl-fundamentals.md
04-reinforcement-learning/docs/02-on-vs-off-policy.md
04-reinforcement-learning/docs/03-algorithm-comparison.md
04-reinforcement-learning/appendix-llamafactory/README.md
04-reinforcement-learning/appendix-llamafactory/configs/dpo_qwen.yaml
04-reinforcement-learning/appendix-llamafactory/configs/grpo_qwen.yaml
04-reinforcement-learning/appendix-llamafactory/configs/ppo_qwen.yaml
04-reinforcement-learning/appendix-llamafactory/run.sh
```

- [ ] **Step 2: Run the reward demo (the one script that needs no GPU/training)**

```bash
cd 04-reinforcement-learning/01-fundamentals && python reward_demo.py
```

Expected: 25-row table of scored responses, no errors.

- [ ] **Step 3: Check all Python files parse without syntax errors**

```bash
python -c "
import ast
from pathlib import Path

scripts = list(Path('04-reinforcement-learning').rglob('*.py'))
for s in scripts:
    ast.parse(s.read_text())
    print(f'  OK: {s}')
print(f'All {len(scripts)} Python files valid.')
"
```

Expected: all files listed as OK.

- [ ] **Step 4: Final commit**

```bash
git add 04-reinforcement-learning/
git commit -m "feat: complete 04-reinforcement-learning module (PPO, DPO, GRPO, LlamaFactory)"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| Standalone folder | Task 1 (`04-reinforcement-learning/`) |
| 01-fundamentals with reward demo | Tasks 2–4 |
| 02-on-policy PPO with reward model | Tasks 5–8 |
| 03-off-policy DPO | Tasks 9–10 |
| 03-off-policy GRPO | Task 11 |
| inference.py in each section | Tasks 7, 12 |
| docs/ (3 concept files) | Task 14 |
| appendix-llamafactory YAML configs | Task 15 |
| appendix README | Task 16 |
| Top-level README | Task 17 |
| English throughout | All tasks |
| Qwen2.5-0.5B | Tasks 6, 10, 11, 15 |
| MPS + GPU device detection | Tasks 5, 6, 7, 10, 11, 12 |
| requirements.txt updated | Task 1 |
| GRPO classification note | Task 11, Task 13 README |
| TRL current import paths | Task 6 (`trl.experimental.ppo`) |
| LlamaFactory stage: ppo/dpo/grpo | Task 15 |
