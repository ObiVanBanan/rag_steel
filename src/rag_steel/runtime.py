"""Runtime protection helpers and controlled failure types."""

from __future__ import annotations

from contextlib import contextmanager
from threading import BoundedSemaphore


class SearchBusyError(RuntimeError):
    """Raised when the search concurrency gate is saturated."""


class EmbeddingTimeoutError(RuntimeError):
    """Raised when OpenAI embedding requests time out."""


class EmbeddingUpstreamError(RuntimeError):
    """Raised when OpenAI embedding requests fail after bounded retries."""


class DeepSeekTimeoutError(RuntimeError):
    """Raised when DeepSeek requests time out."""


class DeepSeekUpstreamError(RuntimeError):
    """Raised when DeepSeek requests fail after bounded retries."""


class DeepSeekConfigurationError(RuntimeError):
    """Raised when DeepSeek is required but not configured."""


class DeepSeekInvalidResponseError(RuntimeError):
    """Raised when DeepSeek returns malformed or empty JSON content."""


class SearchBackendTimeoutError(RuntimeError):
    """Raised when Qdrant search requests time out."""


class SearchBackendUnavailableError(RuntimeError):
    """Raised when Qdrant search requests fail after bounded retries."""


class SearchConcurrencyGate:
    def __init__(self, max_concurrent_searches: int) -> None:
        self._semaphore = BoundedSemaphore(max_concurrent_searches)

    def try_acquire(self) -> bool:
        return self._semaphore.acquire(blocking=False)

    def release(self) -> None:
        self._semaphore.release()

    @contextmanager
    def acquire(self):
        acquired = self.try_acquire()
        if not acquired:
            raise SearchBusyError("Search service is temporarily busy")
        try:
            yield
        finally:
            self.release()


__all__ = [
    "EmbeddingTimeoutError",
    "EmbeddingUpstreamError",
    "DeepSeekConfigurationError",
    "DeepSeekInvalidResponseError",
    "DeepSeekTimeoutError",
    "DeepSeekUpstreamError",
    "SearchBackendTimeoutError",
    "SearchBackendUnavailableError",
    "SearchBusyError",
    "SearchConcurrencyGate",
]
