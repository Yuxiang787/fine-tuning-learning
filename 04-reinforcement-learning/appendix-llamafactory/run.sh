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
