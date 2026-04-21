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
