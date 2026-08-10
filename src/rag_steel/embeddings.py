"""Production embedding interface and OpenAI implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import numpy as np

from rag_steel.settings import Settings


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(slots=True)
class OpenAIEmbedder:
    settings: Settings
    _client: httpx.Client = field(init=False, repr=False)

    embedding_revision: str = ""
    embedding_dtype: str = "float32"
    max_sequence_length: int = 8191

    def __post_init__(self) -> None:
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the production embedder")

        self._client = httpx.Client(
            base_url=self.settings.openai_base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.settings.openai_timeout_seconds,
        )

    @property
    def model_name(self) -> str:
        return self.settings.embedding_model

    @property
    def dimension(self) -> int:
        return self.settings.embedding_dimension

    def _embed_batch(self, batch: list[str], dimensions: int | None) -> np.ndarray:
        payload: dict[str, Any] = {
            "input": batch,
            "model": self.model_name,
            "encoding_format": "float",
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions

        response = self._client.post("embeddings", json=payload)
        response.raise_for_status()
        response_payload = response.json()
        data = response_payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("OpenAI embeddings response is missing data")
        vectors = [item.get("embedding") for item in data]
        try:
            array = np.asarray(vectors, dtype=np.float32)
        except ValueError as exc:
            raise RuntimeError("OpenAI embeddings response contains invalid vector shapes") from exc
        if array.ndim != 2:
            raise RuntimeError(f"Embeddings must be 2D, got shape {array.shape}")
        if array.shape[0] != len(batch):
            raise RuntimeError(
                "OpenAI embeddings response returned "
                f"{array.shape[0]} vectors for {len(batch)} texts"
            )
        if array.shape[1] != self.dimension:
            raise RuntimeError(
                "OpenAI embeddings response returned "
                f"dimension {array.shape[1]}, expected {self.dimension}"
            )
        if not np.isfinite(array).all():
            raise RuntimeError("OpenAI embeddings response contains non-finite values")
        return array

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        requested_dimensions = None
        if self.dimension != 1536:
            requested_dimensions = self.dimension

        vectors: list[list[float]] = []
        batch_size = max(1, self.settings.dense_batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings = self._embed_batch(batch, requested_dimensions)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            embeddings = embeddings / norms
            vectors.extend(embedding.tolist() for embedding in embeddings)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_embedder(settings: Settings) -> Embedder:
    return OpenAIEmbedder(settings)


__all__ = [
    "Embedder",
    "OpenAIEmbedder",
    "create_embedder",
]
