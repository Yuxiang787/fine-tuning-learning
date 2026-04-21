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
