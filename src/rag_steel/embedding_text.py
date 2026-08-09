"""Model-specific embedding text formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass

from rag_steel.config import DEFAULT_MODEL_NAME


@dataclass(slots=True)
class EmbeddingTextAdapter:
    model_name: str = DEFAULT_MODEL_NAME

    def _needs_e5_prefix(self) -> bool:
        model = self.model_name.lower()
        return "multilingual-e5" in model or model.startswith("intfloat/e5")

    def prepare_query(self, text: str) -> str:
        if not text:
            return text
        if self._needs_e5_prefix():
            if text.startswith("query:") or text.startswith("passage:"):
                return text
            return f"query: {text}"
        return text

    def prepare_document(self, text: str) -> str:
        if not text:
            return text
        if self._needs_e5_prefix():
            if text.startswith("passage:") or text.startswith("query:"):
                return text
            return f"passage: {text}"
        return text
