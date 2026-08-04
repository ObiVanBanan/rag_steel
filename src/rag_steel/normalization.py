"""Deterministic normalization helpers for source and LD fields."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

_SPACE_RE = re.compile(r"\s+")
_ARTICLE_STRIP_RE = re.compile(r"[\s./\\\\_-]+")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

_BRAND_ALIASES = {
    "temper": "Temper",
    "broen": "Broen",
    "also": "ALSO",
    "алсо": "ALSO",
    "marshal": "MARSHAL",
    "маршал": "MARSHAL",
    "бивал": "Бивал",
    "bival": "Бивал",
    "forteca": "FORTECA",
}

_CONNECTION_ALIASES = {
    "фланцевое": "фланцевое",
    "фланцевый": "фланцевое",
    "фланец": "фланцевое",
    "flanged": "фланцевое",
    "резьбовое": "резьбовое",
    "резьбовой": "резьбовое",
    "резьба": "резьбовое",
    "threaded": "резьбовое",
    "сварное": "сварное",
    "сварной": "сварное",
    "сварка": "сварное",
    "welded": "сварное",
    "под приварку": "сварное",
    "муфтовое": "муфтовое",
    "муфтовый": "муфтовое",
    "coupling": "муфтовое",
}

_MEDIUM_ALIASES = {
    "жидкость": "жидкость",
    "вода": "жидкость",
    "water": "жидкость",
    "газ": "газ",
    "gas": "газ",
    "пар": "пар",
    "steam": "пар",
    "нефть": "нефть",
    "oil": "нефть",
    "нефтепродукты": "нефтепродукты",
}

_CONTROL_ALIASES = {
    "ручное": "ручное",
    "manual": "ручное",
    "рукоятка": "ручное",
    "рычаг": "ручное",
    "электропривод": "электропривод",
    "electric": "электропривод",
    "пневмопривод": "пневмопривод",
    "pneumatic": "пневмопривод",
    "редуктор": "редуктор",
    "gear": "редуктор",
}


@dataclass(slots=True)
class ArticleNormalization:
    article_raw: str | None
    article_norm: str | None
    article_compact: str | None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def normalize_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = unicodedata.normalize("NFKC", str(value))
    text = text.casefold().replace("ё", "е").replace("\xa0", " ")
    text = _SPACE_RE.sub(" ", text).strip()
    return text or None


def normalize_article(value: Any) -> ArticleNormalization:
    if _is_missing(value):
        return ArticleNormalization(None, None, None)

    article_raw = str(value)
    article_norm = normalize_text(article_raw)
    if article_norm is None:
        return ArticleNormalization(article_raw, None, None)

    article_compact = _ARTICLE_STRIP_RE.sub("", article_norm)
    return ArticleNormalization(article_raw, article_norm, article_compact or None)


def _normalize_by_alias(value: Any, aliases: dict[str, str]) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    for alias, canonical in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return canonical
    return text


def normalize_brand(value: Any) -> str | None:
    return _normalize_by_alias(value, _BRAND_ALIASES)


def normalize_connection(value: Any) -> str | None:
    return _normalize_by_alias(value, _CONNECTION_ALIASES)


def normalize_medium(value: Any) -> str | None:
    return _normalize_by_alias(value, _MEDIUM_ALIASES)


def normalize_control(value: Any) -> str | None:
    return _normalize_by_alias(value, _CONTROL_ALIASES)


def normalize_dn(value: Any) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = normalize_text(value)
    if text is None:
        return None
    match = _NUMBER_RE.search(text.replace(",", "."))
    return float(match.group(0).replace(",", ".")) if match else None


def normalize_pn_bar(value: Any) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = normalize_text(value)
    if text is None:
        return None
    number_match = _NUMBER_RE.search(text.replace(",", "."))
    if not number_match:
        return None

    number = float(number_match.group(0).replace(",", "."))
    if "мпа" in text or "mpa" in text:
        return number * 10.0
    return number


def normalize_temperature(value: Any) -> str | None:
    text = normalize_text(value)
    return text


def normalize_length(value: Any) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = normalize_text(value)
    if text is None:
        return None

    number_match = _NUMBER_RE.search(text.replace(",", "."))
    if not number_match:
        return None

    number = float(number_match.group(0).replace(",", "."))
    if "см" in text or re.search(r"\bcm\b", text):
        return number * 10.0
    if (text.endswith("m") or text.endswith("м")) and "мм" not in text and not text.endswith("mm"):
        return number * 1000.0
    return number


__all__ = [
    "ArticleNormalization",
    "normalize_text",
    "normalize_article",
    "normalize_brand",
    "normalize_connection",
    "normalize_medium",
    "normalize_control",
    "normalize_dn",
    "normalize_pn_bar",
    "normalize_temperature",
    "normalize_length",
]
