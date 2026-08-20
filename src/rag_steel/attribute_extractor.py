"""DeepSeek attribute extractor."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import sleep
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, model_validator

from rag_steel.competitor_registry import COMPETITOR_BRANDS
from rag_steel.normalization import (
    normalize_body_material,
    normalize_connection,
    normalize_control,
    normalize_length,
    normalize_medium,
    normalize_semantic_dn,
    normalize_semantic_pn_bar,
    normalize_supported_brand,
    normalize_temperature,
    normalize_text,
)
from rag_steel.observability import log_deepseek_upstream_failure
from rag_steel.runtime import (
    DeepSeekConfigurationError,
    DeepSeekInvalidResponseError,
    DeepSeekTimeoutError,
    DeepSeekUpstreamError,
)
from rag_steel.settings import Settings


class QueryAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brand: str | None = None
    article: str | None = None

    dn: float | None = None
    pn_bar: float | None = None
    connection: str | None = None

    body_material: str | None = None
    medium: str | None = None
    control: str | None = None
    temperature: str | None = None
    length_mm: float | None = None

    series: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_brand_field(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "brand" not in data and "raw_brand" in data:
            data = dict(data)
            data["brand"] = data.get("raw_brand")
        if "raw_brand" in data:
            data = dict(data)
            data.pop("raw_brand", None)
        return data


def _normalize_raw_fragment(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\xa0", " ").strip()
    text = " ".join(text.split())
    return text or None


def _normalize_extracted_payload(raw_payload: dict[str, Any]) -> QueryAttributes:
    return QueryAttributes(
        brand=normalize_supported_brand(raw_payload.get("brand")),
        article=_normalize_raw_fragment(raw_payload.get("article")),
        dn=normalize_semantic_dn(raw_payload.get("dn")),
        pn_bar=normalize_semantic_pn_bar(raw_payload.get("pn_bar")),
        connection=normalize_connection(raw_payload.get("connection")),
        body_material=normalize_body_material(raw_payload.get("body_material")),
        medium=normalize_medium(raw_payload.get("medium")),
        control=normalize_control(raw_payload.get("control")),
        temperature=normalize_temperature(raw_payload.get("temperature")),
        length_mm=normalize_length(raw_payload.get("length_mm")),
        series=normalize_text(raw_payload.get("series")),
    )


@dataclass(slots=True)
class DeepSeekAttributeExtractor:
    settings: Settings
    _client: httpx.Client | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.settings.deepseek_api_key:
            return

        self._client = httpx.Client(
            base_url=self.settings.deepseek_base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.settings.deepseek_timeout_seconds,
        )

    @staticmethod
    def _json_schema_example() -> dict[str, Any]:
        return {
            "brand": None,
            "article": None,
            "dn": 80,
            "pn_bar": 16,
            "connection": "сварное",
            "body_material": "сталь 09Г2С",
            "medium": "газ",
            "control": "ручное",
            "temperature": None,
            "length_mm": None,
            "series": None,
        }

    @staticmethod
    def _system_prompt() -> str:
        supported_brands = ", ".join(COMPETITOR_BRANDS)
        schema_example = json.dumps(
            DeepSeekAttributeExtractor._json_schema_example(),
            ensure_ascii=False,
            indent=2,
        )
        return (
            "Ты semantic interpreter для запроса к каталогу.\n"
            "Интерпретируй запрос в canonical технические атрибуты, а не копируй текст дословно.\n"
            "Не выбирай товар и не выполняй поиск. Если атрибут отсутствует, верни null.\n"
            "Разрешено исправлять очевидные опечатки и разговорные формы, "
            "если интерпретация однозначна.\n"
            "Если интерпретация неуверенная, верни null только для отсутствующих "
            "атрибутов или неоднозначных текстовых форм.\n\n"
            f"Поддерживаемые бренды: {supported_brands}.\n"
            "brand:\n"
            "- возвращай только canonical supported competitor brand из списка выше;\n"
            "- поддерживай русские написания и очевидные опечатки;\n"
            "- если бренд не относится к списку выше, верни null;\n"
            "- примеры: 'Темпер' -> 'Temper', 'темпр' -> 'Temper', 'Маршал' -> 'MARSHAL',\n"
            "  'Броен' -> 'Broen', 'алсо' -> 'ALSO', 'фортека' -> 'FORTECA',\n"
            "  'Valtec' -> null.\n\n"
            "article:\n"
            "- если в запросе явно присутствует артикул, каталожный номер или "
            "идентификатор товара,\n"
            "  верни его буквально;\n"
            "- не угадывай отсутствующий артикул;\n"
            "- не меняй символы, регистр, пробелы и дефисы.\n\n"
            "dn:\n"
            "- интерпретируй наиболее вероятный стандартный DN;\n"
            "- поддерживай формы 'ду 50', 'DN50', 'пятидесятый', 'сотка', 'ду сто';\n"
            "- исправляй только очевидные near-standard typos, например 'DN51' -> 50, "
            "'DN57' -> 50, 'DN64' -> 65 и 'DN107' -> 100;\n"
            "- если correction неочевидна, не угадывай дальний стандарт;\n"
            "- возвращай число в миллиметрах.\n\n"
            "pn_bar:\n"
            "- интерпретируй PN семантически;\n"
            "- поддерживай 'РУ16', 'PN16', '16 бар', '1.6 МПа', 'ру двадцать пять';\n"
            "- не меняй бизнес-семантику: PN16 не превращай в PN25;\n"
            "- возвращай число в bar.\n\n"
            "connection:\n"
            "- возвращай canonical connection terminology проекта;\n"
            "- примеры: 'фланец', 'фланцевый', 'на фланцах' -> 'фланцевое';\n"
            "  'резьба', 'резьбовой' -> 'резьбовое';\n"
            "  'под сварку', 'сварной' -> 'сварное'.\n\n"
            "soft attributes:\n"
            "- нормализуй body material, medium, control, temperature, length и series, "
            "если это ясно;\n"
            "- предпочитай canonical technical wording.\n\n"
            "Верни только json по схеме ниже.\n\n"
            "Схема и пример:\n"
            f"{schema_example}"
        )

    def _request_payload(self, query: str) -> dict[str, Any]:
        return {
            "model": self.settings.deepseek_model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": query},
            ],
            "temperature": 0,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "max_tokens": 512,
        }

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code in {429, 500, 502, 503, 504}

    @staticmethod
    def _retry_delay_seconds(*, attempt: int, base_delay_seconds: float) -> float:
        return max(0.0, base_delay_seconds) * (2 ** max(0, attempt - 1))

    @staticmethod
    def _log_upstream_failure(
        *,
        attempt: int,
        error_type: str,
        status_code: int | None,
        request_url: str | None,
        retryable: bool,
        exc: Exception,
    ) -> None:
        log_deepseek_upstream_failure(
            upstream="deepseek",
            error_type=error_type,
            status_code=status_code,
            request_url=request_url,
            retryable=retryable,
            attempt=attempt,
            exception_type=type(exc).__name__,
        )

    def _post_completion(self, payload: dict[str, Any]) -> httpx.Response:
        if self._client is None:
            raise DeepSeekConfigurationError(
                "DEEPSEEK_API_KEY is required for V5 semantic attribute extraction"
            )

        max_attempts = max(1, self.settings.upstream_max_attempts)
        base_delay_seconds = max(0.0, self.settings.upstream_retry_base_delay_seconds)
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.post("chat/completions", json=payload)
                response.raise_for_status()
                return response
            except httpx.TimeoutException as exc:
                request_url = str(exc.request.url) if getattr(exc, "request", None) else None
                self._log_upstream_failure(
                    attempt=attempt,
                    error_type="timeout",
                    status_code=None,
                    request_url=request_url,
                    retryable=True,
                    exc=exc,
                )
                last_error = exc
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                request_url = str(exc.request.url) if getattr(exc, "request", None) else None
                retryable = self._is_retryable_status(status_code)
                self._log_upstream_failure(
                    attempt=attempt,
                    error_type="upstream",
                    status_code=status_code,
                    request_url=request_url,
                    retryable=retryable,
                    exc=exc,
                )
                if not retryable:
                    raise DeepSeekUpstreamError(
                        f"DeepSeek request failed with status {status_code}"
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
                request_url = str(exc.request.url) if getattr(exc, "request", None) else None
                self._log_upstream_failure(
                    attempt=attempt,
                    error_type="upstream",
                    status_code=None,
                    request_url=request_url,
                    retryable=True,
                    exc=exc,
                )
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
        raise DeepSeekInvalidResponseError("DeepSeek response does not contain JSON content")

    @staticmethod
    def _coerce_payload(raw_content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise DeepSeekInvalidResponseError("DeepSeek response is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise DeepSeekInvalidResponseError("DeepSeek response JSON must be an object")
        return parsed

    def extract(self, query: str) -> QueryAttributes:
        response = self._post_completion(self._request_payload(query))
        payload = response.json()
        raw_content = self._extract_content(payload)
        parsed = self._coerce_payload(raw_content)
        return _normalize_extracted_payload(parsed)


def create_attribute_extractor(settings: Settings) -> DeepSeekAttributeExtractor:
    return DeepSeekAttributeExtractor(settings)


__all__ = [
    "DeepSeekAttributeExtractor",
    "QueryAttributes",
    "create_attribute_extractor",
]

ExtractedAttributes = QueryAttributes
