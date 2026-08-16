"""DeepSeek attribute extractor."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import sleep
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from rag_steel.normalization import (
    normalize_body_material,
    normalize_connection,
    normalize_control,
    normalize_dn,
    normalize_length,
    normalize_medium,
    normalize_pn_bar,
    normalize_temperature,
    normalize_text,
)
from rag_steel.runtime import DeepSeekTimeoutError, DeepSeekUpstreamError
from rag_steel.settings import Settings


class ExtractedAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dn: float | None = None
    pn_bar: float | None = None
    connection: str | None = None

    body_material: str | None = None
    medium: str | None = None
    control: str | None = None
    temperature: str | None = None
    length_mm: float | None = None

    series: str | None = None
    article: str | None = None


def _normalize_extracted_payload(raw_payload: dict[str, Any]) -> ExtractedAttributes:
    return ExtractedAttributes(
        dn=normalize_dn(raw_payload.get("dn")),
        pn_bar=normalize_pn_bar(raw_payload.get("pn_bar")),
        connection=normalize_connection(raw_payload.get("connection")),
        body_material=normalize_body_material(raw_payload.get("body_material")),
        medium=normalize_medium(raw_payload.get("medium")),
        control=normalize_control(raw_payload.get("control")),
        temperature=normalize_temperature(raw_payload.get("temperature")),
        length_mm=normalize_length(raw_payload.get("length_mm")),
        series=normalize_text(raw_payload.get("series")),
        article=normalize_text(raw_payload.get("article")),
    )


@dataclass(slots=True)
class DeepSeekAttributeExtractor:
    settings: Settings
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek extraction")

        self._client = httpx.Client(
            base_url=self.settings.deepseek_base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.settings.deepseek_timeout_seconds,
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "Извлекай только характеристики, явно присутствующие в запросе. "
            "Не подбирай значения самостоятельно. "
            "Не ищи аналог. Не выбирай товар. "
            "Если параметр отсутствует - null. "
            "Верни только JSON согласно схеме."
        )

    def _request_payload(self, query: str) -> dict[str, Any]:
        return {
            "model": self.settings.deepseek_model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": query},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code in {429, 500, 502, 503, 504}

    @staticmethod
    def _retry_delay_seconds(*, attempt: int, base_delay_seconds: float) -> float:
        return max(0.0, base_delay_seconds) * (2 ** max(0, attempt - 1))

    def _post_completion(self, payload: dict[str, Any]) -> httpx.Response:
        max_attempts = max(1, self.settings.upstream_max_attempts)
        base_delay_seconds = max(0.0, self.settings.upstream_retry_base_delay_seconds)
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.post("chat/completions", json=payload)
                response.raise_for_status()
                return response
            except httpx.TimeoutException as exc:
                last_error = exc
            except httpx.HTTPStatusError as exc:
                if not self._is_retryable_status(exc.response.status_code):
                    raise DeepSeekUpstreamError(
                        f"DeepSeek request failed with status {exc.response.status_code}"
                    ) from exc
                last_error = exc
                if attempt >= max_attempts:
                    break
                sleep(
                    self._retry_delay_seconds(
                        attempt=attempt,
                        base_delay_seconds=base_delay_seconds,
                    )
                )
                continue
            except httpx.RequestError as exc:
                last_error = exc
            except Exception as exc:
                raise DeepSeekUpstreamError("DeepSeek request failed") from exc

            if attempt >= max_attempts:
                break

            sleep(
                self._retry_delay_seconds(
                    attempt=attempt,
                    base_delay_seconds=base_delay_seconds,
                )
            )

        if isinstance(last_error, httpx.TimeoutException):
            raise DeepSeekTimeoutError("DeepSeek request timed out") from last_error
        raise DeepSeekUpstreamError("DeepSeek request failed") from last_error

    @staticmethod
    def _extract_content(response_payload: dict[str, Any]) -> str:
        choices = response_payload.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message") or {}
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, dict):
                        return json.dumps(content, ensure_ascii=False)
        content = response_payload.get("content")
        if isinstance(content, str):
            return content
        raise RuntimeError("DeepSeek response does not contain JSON content")

    @staticmethod
    def _coerce_payload(raw_content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("DeepSeek response is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("DeepSeek response JSON must be an object")
        return parsed

    def extract(self, query: str) -> ExtractedAttributes:
        response = self._post_completion(self._request_payload(query))
        payload = response.json()
        raw_content = self._extract_content(payload)
        parsed = self._coerce_payload(raw_content)
        return _normalize_extracted_payload(parsed)


def create_attribute_extractor(settings: Settings) -> DeepSeekAttributeExtractor | None:
    if not settings.deepseek_api_key:
        return None
    return DeepSeekAttributeExtractor(settings)


__all__ = [
    "DeepSeekAttributeExtractor",
    "ExtractedAttributes",
    "create_attribute_extractor",
]
