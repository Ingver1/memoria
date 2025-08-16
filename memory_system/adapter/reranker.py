"""
Preference-based reranking utilities.

This module defines :class:`PreferenceReranker` which loads a base cross-encoder
model and optional LoRA adapters produced by the training script. The class
provides a simple ``rerank`` method that scores documents against a query and
returns them ordered by preference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used for typing only
    from collections.abc import Iterable

try:  # pragma: no cover - optional runtime dependency
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    msg = "torch is required to use PreferenceReranker. Install torch or ai-memory[full]."
    raise ModuleNotFoundError(msg) from exc

try:  # pragma: no cover - optional runtime dependency
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    msg = "transformers is required to use PreferenceReranker. Install ai-memory[rerank]."
    raise ModuleNotFoundError(msg) from exc

try:  # pragma: no cover - optional runtime dependency
    from peft import PeftModel
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    msg = "peft is required to use PreferenceReranker. Install ai-memory[rerank]."
    raise ModuleNotFoundError(msg) from exc


@dataclass
class RerankResult:
    """Container for a document and its score."""

    document: str
    score: float


class PreferenceReranker:
    """
    Cross-encoder reranker fine-tuned with preference optimization.

    Parameters
    ----------
    base_model:
        Hugging Face model name used as the base cross-encoder.
    adapter_path:
        Optional path containing LoRA weights produced by the training script.
        If provided, the adapter is loaded on top of the base model.
    device:
        Optional device identifier. Defaults to ``cuda`` when available or
        ``cpu`` otherwise.

    """

    def __init__(
        self,
        base_model: str = "distilbert-base-uncased",
        adapter_path: str | None = None,
        device: str | None = None,
    ) -> None:
        """Load the base model and optional LoRA adapter."""
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model)
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # ------------------------------------------------------------------
    def score(self, query: str, document: str) -> float:
        """Return the model score for ``document`` given ``query``."""
        inputs = self.tokenizer(
            query,
            document,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze()
        return float(logits.item())

    # ------------------------------------------------------------------
    def rerank(self, query: str, documents: Iterable[str]) -> list[RerankResult]:
        """
        Return ``documents`` scored and ordered by relevance.

        Parameters
        ----------
        query:
            Query string to compare documents against.
        documents:
            Iterable of document strings to score.

        Returns
        -------
        list of :class:`RerankResult`
            Documents ordered from most to least relevant according to the
            model.

        """
        results = [RerankResult(doc, self.score(query, doc)) for doc in documents]
        results.sort(key=lambda r: r.score, reverse=True)
        return results
