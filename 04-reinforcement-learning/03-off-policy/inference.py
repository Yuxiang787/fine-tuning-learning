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
