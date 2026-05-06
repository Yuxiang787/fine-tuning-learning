import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

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


class _PassthroughBackbone(nn.Module):
    """
    Fake backbone for TRLCompatibleSentimentReward.

    TRL's get_reward() calls backbone(input_ids=...) and then model.score(hidden_states).
    We capture the input_ids here so score() can decode them for DistilBERT.
    """

    def __init__(self):
        super().__init__()
        self.stored_input_ids: torch.Tensor | None = None
        # A dummy parameter so .to(device) / accelerate work on this module
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,
    ) -> BaseModelOutput:
        self.stored_input_ids = input_ids
        batch, seq_len = input_ids.shape
        # Return dummy hidden states (hidden_size=1) — shape is all get_reward uses
        dummy = self._dummy.new_zeros(batch, seq_len, 1)
        return BaseModelOutput(last_hidden_state=dummy, hidden_states=(dummy,))


class TRLCompatibleSentimentReward(nn.Module):
    """
    DistilBERT SST-2 reward model with the interface TRL's experimental PPOTrainer needs.

    TRL's get_reward() requires:
      model.base_model_prefix  → name of the backbone submodule
      model.<prefix>(input_ids, ...)  → runs the backbone, returns hidden states
      model.score(hidden_states)  → (batch, seq_len, 1) scalar per token position

    Because TRL feeds Qwen2 token IDs into the backbone, we use a passthrough
    backbone that just stores those IDs, then score() decodes them to text and
    re-encodes with DistilBERT's tokenizer for sentiment scoring.
    """

    base_model_prefix = "backbone"

    def __init__(self, policy_tokenizer: AutoTokenizer):
        super().__init__()
        self.backbone = _PassthroughBackbone()
        # Keep classifier on CPU — MPS can struggle with DistilBERT inference,
        # and the overhead of CPU scoring is small compared to Qwen2 generation.
        self._classifier = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL_NAME
        ).eval()
        self._reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
        self._policy_tokenizer = policy_tokenizer

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Called by TRL's get_reward with (batch, seq_len, 1) dummy hidden states.
        Decodes the stored Qwen2 input_ids → text → DistilBERT sentiment score.
        Returns (batch, seq_len, 1) with the same scalar broadcast over all positions;
        TRL slices at the EOS position to get the per-sequence reward.
        """
        input_ids = self.backbone.stored_input_ids
        texts = self._policy_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        enc = self._reward_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self._classifier(**enc).logits  # run on CPU
        pos_prob = logits.softmax(dim=-1)[:, 1]  # (batch,) in [0, 1]
        seq_len = hidden_states.shape[1]
        return pos_prob[:, None, None].expand(-1, seq_len, 1).to(hidden_states.device)


def load_reward_model(device: str) -> SentimentRewardModel:
    model = SentimentRewardModel().to(device)
    model.eval()
    return model


def load_trl_reward_model(
    policy_tokenizer: AutoTokenizer, device: str
) -> TRLCompatibleSentimentReward:
    model = TRLCompatibleSentimentReward(policy_tokenizer)
    model.backbone.to(device)
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
