"""
Train a preference-based reranker with LoRA and DPO.

The script expects pairs of `(better, worse)` documents for each query and
optimizes a cross-encoder head using a Direct Preference Optimisation objective.
Only LoRA adapter weights are stored which keeps the repository lightweight.

Example usage::

    python scripts/train_preference_reranker.py --dataset pairs.jsonl \
        --output_dir memory_system/adapter/preference_reranker

If ``--synthetic`` is passed a tiny synthetic dataset is used instead. This is
handy for quick sanity checks or CI jobs.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PAIR = tuple[str, str, str]  # query, better, worse


def load_pairs(path: Path) -> list[PAIR]:
    """Load training pairs from a JSONL file."""
    pairs: list[PAIR] = []
    with path.open() as f:
        for line in f:
            data = json.loads(line)
            pairs.append((data["query"], data["better"], data["worse"]))
    return pairs


def synthetic_pairs() -> list[PAIR]:
    """Return a very small synthetic dataset."""
    return [
        (
            "What color is the sky?",
            "The sky is blue.",
            "The sky is green.",
        ),
        (
            "What is 2 + 2?",
            "2 + 2 equals 4.",
            "2 + 2 equals 5.",
        ),
    ]


def dpo_loss(preferred: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    """Direct Preference Optimisation loss."""
    return -torch.nn.functional.logsigmoid(preferred - rejected).mean()


def train(
    model_name: str,
    pairs: Iterable[PAIR],
    output_dir: Path,
    epochs: int = 1,
    lr: float = 5e-5,
) -> None:
    """Run a tiny training loop and save the resulting adapter."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="SEQ_CLS")
    model = get_peft_model(model, lora)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    loader = DataLoader(list(pairs), batch_size=1, shuffle=True)
    for _ in range(epochs):
        for query, better, worse in loader:
            better_inputs = tokenizer(
                query, better, return_tensors="pt", padding=True, truncation=True
            )
            worse_inputs = tokenizer(
                query, worse, return_tensors="pt", padding=True, truncation=True
            )
            better_score = model(**better_inputs).logits.squeeze(-1)
            worse_score = model(**worse_inputs).logits.squeeze(-1)
            loss = dpo_loss(better_score, worse_score)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="JSONL with query/better/worse fields")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("memory_system/adapter/preference_reranker"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--synthetic", action="store_true", help="Use a tiny synthetic dataset")
    args = parser.parse_args()

    if args.synthetic:
        pairs = synthetic_pairs()
    elif args.dataset:
        pairs = load_pairs(args.dataset)
    else:
        raise SystemExit("Either --dataset or --synthetic must be provided")

    train(args.model_name, pairs, args.output_dir, args.epochs, args.lr)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
