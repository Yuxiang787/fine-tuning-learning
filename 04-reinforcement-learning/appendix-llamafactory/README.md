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
