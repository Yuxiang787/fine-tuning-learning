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
