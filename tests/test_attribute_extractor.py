from __future__ import annotations

import importlib
import json
import os
import sys

import httpx
import pytest

import rag_steel.attribute_extractor as attribute_extractor_mod
from rag_steel.runtime import (
    DeepSeekConfigurationError,
    DeepSeekInvalidResponseError,
    DeepSeekTimeoutError,
    DeepSeekUpstreamError,
)


def _reload_settings():
    os.environ["RAG_STEEL_DISABLE_DOTENV"] = "1"
    sys.modules.pop("rag_steel.settings", None)
    return importlib.import_module("rag_steel.settings")


def _load_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("UPSTREAM_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("UPSTREAM_RETRY_BASE_DELAY_SECONDS", "0")
    return _reload_settings().get_settings()


def test_deepseek_extractor_normalizes_model_output(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _load_settings(monkeypatch)
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "brand": "Tempr",
                                    "article": "A-0486",
                                    "dn": 50,
                                    "pn_bar": "1,6 МПа",
                                    "connection": "под приварку",
                                    "body_material": "сталь 09Г2С",
                                    "medium": "Газ",
                                    "control": "manual",
                                    "temperature": "до +80",
                                    "length_mm": 300,
                                    "series": "60",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

        def post(self, path: str, json: dict[str, object]) -> FakeResponse:
            captured["path"] = path
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(attribute_extractor_mod.httpx, "Client", FakeClient)

    extractor = attribute_extractor_mod.create_attribute_extractor(settings)
    assert extractor is not None

    result = extractor.extract("Temper DN50 PN16 для газа")

    assert captured["path"] == "chat/completions"
    assert result.brand == "Temper"
    assert result.article == "A-0486"
    assert result.dn == 50
    assert result.pn_bar == 16
    assert result.connection == "сварное"
    assert result.body_material == "сталь 09г2с"
    assert result.medium == "газ"
    assert result.control == "ручное"
    assert result.length_mm == 300
    assert result.series == "60"


def test_deepseek_system_prompt_requires_brand_in_mixed_queries() -> None:
    prompt = attribute_extractor_mod.DeepSeekAttributeExtractor._system_prompt()

    assert "semantic interpreter" in prompt
    assert "Поддерживаемые бренды:" in prompt
    assert "Temper" in prompt
    assert "Valtec' -> null" in prompt
    assert "DN51" in prompt
    assert "DN57" in prompt
    assert "DN64" in prompt
    assert "DN107" in prompt
    assert "PN16" in prompt
    assert "1.6 МПа" in prompt
    assert "на фланцах" in prompt
    assert "под сварку" in prompt
    assert "canonical technical wording" in prompt


def test_deepseek_extractor_interprets_semantic_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _load_settings(monkeypatch)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "brand": "Маршал",
                                    "article": "A-0486",
                                    "dn": "ду 64",
                                    "pn_bar": "ру двадцать пять",
                                    "connection": "на фланцах",
                                    "body_material": "нержавеющая сталь",
                                    "medium": "вода",
                                    "control": "manual",
                                    "temperature": "до +80",
                                    "length_mm": "30 см",
                                    "series": "60",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, **_: object) -> None:
            return None

        def post(self, path: str, json: dict[str, object]) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(attribute_extractor_mod.httpx, "Client", FakeClient)

    extractor = attribute_extractor_mod.create_attribute_extractor(settings)
    result = extractor.extract("Маршал ду64 ру25 на фланцах")

    assert result.brand == "MARSHAL"
    assert result.dn == 65
    assert result.pn_bar == 25
    assert result.connection == "фланцевое"
    assert result.length_mm == 300


def test_deepseek_extractor_drops_unsupported_brand(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _load_settings(monkeypatch)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {"message": {"content": json.dumps({"brand": "Valtec"}, ensure_ascii=False)}}
                ]
            }

    class FakeClient:
        def __init__(self, **_: object) -> None:
            return None

        def post(self, path: str, json: dict[str, object]) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(attribute_extractor_mod.httpx, "Client", FakeClient)

    extractor = attribute_extractor_mod.create_attribute_extractor(settings)
    result = extractor.extract("Valtec DN50")

    assert result.brand is None


def test_deepseek_extractor_rejects_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _load_settings(monkeypatch)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "{not-json}"}}]}

    class FakeClient:
        def __init__(self, **_: object) -> None:
            return None

        def post(self, path: str, json: dict[str, object]) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(attribute_extractor_mod.httpx, "Client", FakeClient)

    extractor = attribute_extractor_mod.create_attribute_extractor(settings)
    assert extractor is not None

    with pytest.raises(DeepSeekInvalidResponseError, match="valid JSON"):
        extractor.extract("Temper DN50 PN16")


def test_deepseek_extractor_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("UPSTREAM_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("UPSTREAM_RETRY_BASE_DELAY_SECONDS", "0")
    settings = _reload_settings().get_settings()

    extractor = attribute_extractor_mod.create_attribute_extractor(settings)

    with pytest.raises(DeepSeekConfigurationError, match="required"):
        extractor.extract("Temper DN50 PN16")


def test_deepseek_extractor_raises_upstream_error_on_5xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _load_settings(monkeypatch)
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    sleep_calls: list[float] = []

    class FakeResponse:
        def __init__(self) -> None:
            self.status_code = 503
            self.request = request

        def raise_for_status(self) -> None:
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError(
                "service unavailable",
                request=request,
                response=response,
            )

        def json(self) -> dict[str, object]:
            return {}

    class FakeClient:
        def __init__(self, **_: object) -> None:
            return None

        def post(self, path: str, json: dict[str, object]) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(attribute_extractor_mod.httpx, "Client", FakeClient)
    monkeypatch.setattr(
        attribute_extractor_mod,
        "sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    extractor = attribute_extractor_mod.create_attribute_extractor(settings)
    assert extractor is not None

    with pytest.raises(DeepSeekUpstreamError):
        extractor.extract("Temper DN50 PN16")

    assert sleep_calls == [0.0]


def test_deepseek_extractor_raises_timeout_after_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _load_settings(monkeypatch)
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    sleep_calls: list[float] = []

    class FakeClient:
        def __init__(self, **_: object) -> None:
            self.calls = 0

        def post(self, path: str, json: dict[str, object]) -> httpx.Response:
            self.calls += 1
            raise httpx.ReadTimeout("timed out", request=request)

    monkeypatch.setattr(attribute_extractor_mod.httpx, "Client", FakeClient)
    monkeypatch.setattr(
        attribute_extractor_mod,
        "sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    extractor = attribute_extractor_mod.create_attribute_extractor(settings)
    assert extractor is not None

    with pytest.raises(DeepSeekTimeoutError):
        extractor.extract("Temper DN50 PN16")

    assert sleep_calls == [0.0]
