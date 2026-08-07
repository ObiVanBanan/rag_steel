"""Unified query normalization for the hybrid LD search pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag_steel.config import DEFAULT_MODEL_NAME
from rag_steel.normalization import (
    normalize_article,
    normalize_brand,
    normalize_connection,
    normalize_control,
    normalize_dn,
    normalize_medium,
    normalize_pn_bar,
    normalize_text,
)

_SPACE_RE = re.compile(r"\s+")

_BRAND_VALUES = {"Temper", "Broen", "ALSO", "MARSHAL", "Бивал", "FORTECA"}
_CONNECTION_VALUES = {"фланцевое", "резьбовое", "сварное", "муфтовое"}
_MEDIUM_VALUES = {"жидкость", "газ", "пар", "нефть", "нефтепродукты"}
_CONTROL_VALUES = {"ручное", "электропривод", "пневмопривод", "редуктор"}
class ProcessedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str
    normalized: str
    compact: str
    semantic_text: str
    lexical_text: str
    possible_article_tokens: list[str] = Field(default_factory=list)
    brand: str | None = None
    dn: float | None = None
    pn_bar: float | None = None
    connection: str | None = None
    medium: str | None = None
    control: str | None = None


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


def _dedupe_text_parts(parts: list[str | None]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for part in parts:
        if part is None:
            continue
        normalized = " ".join(str(part).split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _format_number(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:g}"


def _titlecase_brand(text: str | None) -> str | None:
    if text is None:
        return None
    if text.isupper():
        return text
    return text[:1].upper() + text[1:]


def _collect_article_tokens(raw: str, normalized: str) -> list[str]:
    tokens: list[str] = []
    for token in re.split(r"\s+", raw):
        if not token:
            continue
        article = normalize_article(token)
        has_article_shape = bool(re.search(r"[./_-]", token)) or (
            len(token) >= 5 and bool(re.search(r"\d", token))
        )
        if has_article_shape:
            tokens.extend(
                candidate
                for candidate in [
                    article.article_raw,
                    article.article_norm,
                    article.article_compact,
                ]
                if candidate
            )

    return _dedupe_text_parts(tokens)


def _extract_query_traits(text: str) -> dict[str, Any]:
    tokens = text.split()
    brand: str | None = None
    for token in tokens:
        if re.search(r"[\d./_-]", token):
            continue
        candidate = _titlecase_brand(normalize_brand(token))
        if candidate in _BRAND_VALUES:
            brand = candidate
            break

    dn: float | None = None
    pn_bar: float | None = None
    connection: str | None = None
    medium: str | None = None
    control: str | None = None

    for token in tokens:
        if re.search(r"[\d./_-]", token):
            continue
        if dn is None and re.search(r"(?i)\b(?:dn|ду|диаметр)", token):
            dn = normalize_dn(token)
        if pn_bar is None and re.search(r"(?i)\b(?:pn|ru|ру|mpa|бар)", token):
            pn_bar = normalize_pn_bar(token)
        if connection is None:
            candidate = normalize_connection(token)
            if candidate in _CONNECTION_VALUES:
                connection = candidate
        if medium is None:
            candidate = normalize_medium(token)
            if candidate in _MEDIUM_VALUES:
                medium = candidate
        if control is None:
            candidate = normalize_control(token)
            if candidate in _CONTROL_VALUES:
                control = candidate

    if dn is None:
        for pattern in (r"(?i)\b(?:dn|ду|диаметр)\s*([0-9]+(?:[.,][0-9]+)?)",):
            match = re.search(pattern, text)
            if match:
                dn = normalize_dn(match.group(1))
                break

    if pn_bar is None:
        for pattern in (r"(?i)\b(?:pn|ru|ру|mpa|бар)\s*([0-9]+(?:[.,][0-9]+)?)",):
            match = re.search(pattern, text)
            if match:
                pn_bar = normalize_pn_bar(match.group(1))
                break

    return {
        "brand": brand,
        "dn": dn,
        "pn_bar": pn_bar,
        "connection": connection,
        "medium": medium,
        "control": control,
    }


class QueryProcessor:
    """Convert a raw user query into search-ready text features."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self.embedding_adapter = EmbeddingTextAdapter(model_name=model_name)

    def process(self, query: str) -> ProcessedQuery:
        raw = query or ""
        normalized = normalize_text(raw) or ""
        compact = normalized.replace(" ", "")
        traits = _extract_query_traits(normalized)
        possible_article_tokens = _collect_article_tokens(raw, normalized)

        dn_text = _format_number(traits["dn"])
        pn_text = _format_number(traits["pn_bar"])
        main_phrase = " ".join(
            part
            for part in [
                _titlecase_brand(traits["brand"]),
                possible_article_tokens[0] if possible_article_tokens else None,
            ]
            if part
        ).strip()
        if not main_phrase:
            main_phrase = raw.strip() or normalized

        semantic_tail = _dedupe_text_parts(
            [
                f"DN {dn_text}" if dn_text else None,
                f"PN {pn_text}" if pn_text else None,
                f"Соединение: {traits['connection']}" if traits["connection"] else None,
                f"Среда: {traits['medium']}" if traits["medium"] else None,
                f"Управление: {traits['control']}" if traits["control"] else None,
            ]
        )
        semantic_text = ", ".join([main_phrase, *semantic_tail]) if semantic_tail else main_phrase

        lexical_parts = _dedupe_text_parts(
            [
                normalized,
                *possible_article_tokens,
                traits["brand"],
                f"dn{dn_text}" if dn_text else None,
                f"dn {dn_text}" if dn_text else None,
                f"ду{dn_text}" if dn_text else None,
                f"ду {dn_text}" if dn_text else None,
                f"pn{pn_text}" if pn_text else None,
                f"pn {pn_text}" if pn_text else None,
                f"ру{pn_text}" if pn_text else None,
                f"ру {pn_text}" if pn_text else None,
                f"{pn_text} бар" if pn_text else None,
                traits["connection"],
                traits["medium"],
                traits["control"],
            ]
        )
        lexical_text = " ".join(lexical_parts)

        return ProcessedQuery(
            raw=raw,
            normalized=normalized,
            compact=compact,
            semantic_text=semantic_text,
            lexical_text=lexical_text,
            possible_article_tokens=possible_article_tokens,
            brand=traits["brand"],
            dn=traits["dn"],
            pn_bar=traits["pn_bar"],
            connection=traits["connection"],
            medium=traits["medium"],
            control=traits["control"],
        )


def parse_query(query: str, model_name: str = DEFAULT_MODEL_NAME) -> ProcessedQuery:
    return QueryProcessor(model_name=model_name).process(query)
