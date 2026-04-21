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
