#!/usr/bin/env bash
# LlamaFactory RL training — one command per method
# Each command is equivalent to the corresponding Python script in 02-on-policy/ or 03-off-policy/
#
# Prerequisites:
#   pip install llamafactory
#   "$LF_CLI" train --help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LF_CLI="$SCRIPT_DIR/.venv/bin/llamafactory-cli"

echo "=== LlamaFactory RL Training ==="
echo "Choose a method to train:"
echo "  1) PPO  (on-policy)  — requires a pre-trained reward model"
echo "  2) DPO  (off-policy) — requires preference pairs dataset"
echo "  3) GRPO (off-policy) — requires prompts + verifiable reward"
echo ""

case "${1:-}" in
  ppo)
    echo "Running PPO training..."
    "$LF_CLI" train "$SCRIPT_DIR/configs/ppo_qwen.yaml"
    ;;
  dpo)
    echo "Running DPO training..."
    "$LF_CLI" train "$SCRIPT_DIR/configs/dpo_qwen.yaml"
    ;;
  grpo)
    echo "Running GRPO training..."
    "$LF_CLI" train "$SCRIPT_DIR/configs/grpo_qwen.yaml"
    ;;
  all)
    echo "Running DPO then GRPO (skipping PPO — needs reward model first)..."
    "$LF_CLI" train "$SCRIPT_DIR/configs/dpo_qwen.yaml"
    "$LF_CLI" train "$SCRIPT_DIR/configs/grpo_qwen.yaml"
    ;;
  *)
    echo "Usage: ./run.sh [ppo|dpo|grpo|all]"
    echo ""
    echo "Or run directly:"
    echo "  "$LF_CLI" train configs/ppo_qwen.yaml"
    echo "  "$LF_CLI" train configs/dpo_qwen.yaml"
    echo "  "$LF_CLI" train configs/grpo_qwen.yaml"
    exit 1
    ;;
esac

echo "Done. Check outputs/ for saved checkpoints."
