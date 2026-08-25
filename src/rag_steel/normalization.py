"""Deterministic normalization helpers for source and LD fields."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from rag_steel.competitor_registry import COMPETITOR_BRANDS

_SPACE_RE = re.compile(r"\s+")
_ARTICLE_STRIP_RE = re.compile(r"[\s./\\\\_-]+")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

_BRAND_ALIASES = {
    alias: canonical
    for canonical, aliases in COMPETITOR_BRANDS.items()
    for alias in (canonical.lower(), canonical, *aliases)
}

_CONNECTION_ALIASES = {
    "фланцевое": "фланцевое",
    "фланцевый": "фланцевое",
    "фланец": "фланцевое",
    "на фланцах": "фланцевое",
    "flanged": "фланцевое",
    "резьбовое": "резьбовое",
    "резьбовой": "резьбовое",
    "резьба": "резьбовое",
    "threaded": "резьбовое",
    "сварное": "сварное",
    "сварной": "сварное",
    "сварка": "сварное",
    "под сварку": "сварное",
    "welded": "сварное",
    "под приварку": "сварное",
    "приварное": "сварное",
    "приварной": "сварное",
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


def _damerau_distance_at_most_one(left: str, right: str) -> int | None:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > 1:
        return None
    if len(left) == len(right):
        diffs = [
            index for index, (lhs, rhs) in enumerate(zip(left, right, strict=True)) if lhs != rhs
        ]
        if len(diffs) == 1:
            return 1
        if (
            len(diffs) == 2
            and diffs[1] == diffs[0] + 1
            and left[diffs[0]] == right[diffs[1]]
            and left[diffs[1]] == right[diffs[0]]
        ):
            return 1
        return None

    if len(left) > len(right):
        left, right = right, left

    i = j = 0
    edits = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return None
        j += 1

    if j < len(right) or i < len(left):
        edits += 1
    return edits if edits <= 1 else None


def normalize_supported_brand(value: Any) -> str | None:
    brand = normalize_brand(value)
    if brand in COMPETITOR_BRANDS:
        return brand
    normalized = normalize_text(value)
    if normalized is None or len(normalized) < 4:
        return None

    best_distance: int | None = None
    best_candidates: list[str] = []
    for canonical, aliases in COMPETITOR_BRANDS.items():
        candidate_distance: int | None = None
        for alias in (canonical.lower(), canonical, *aliases):
            distance = _damerau_distance_at_most_one(normalized, normalize_text(alias) or alias)
            if distance is None:
                continue
            if candidate_distance is None or distance < candidate_distance:
                candidate_distance = distance
            if candidate_distance == 0:
                break
        if candidate_distance is None:
            continue
        if best_distance is None or candidate_distance < best_distance:
            best_distance = candidate_distance
            best_candidates = [canonical]
        elif candidate_distance == best_distance:
            best_candidates.append(canonical)

    if best_distance is None or best_distance > 1 or len(best_candidates) != 1:
        return None
    return best_candidates[0]


def normalize_connection(value: Any) -> str | None:
    return _normalize_by_alias(value, _CONNECTION_ALIASES)


def normalize_medium(value: Any) -> str | None:
    return _normalize_by_alias(value, _MEDIUM_ALIASES)


def normalize_control(value: Any) -> str | None:
    return _normalize_by_alias(value, _CONTROL_ALIASES)


def normalize_body_material(value: Any) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None

    if "латун" in text:
        return "латунь"

    if "нержавеющ" in text:
        return "нержавеющая сталь"

    match = re.search(r"\b(?:steel|сталь)?\s*([0-9]{2,4}[а-яa-z0-9\-]*)\b", text, re.IGNORECASE)
    if match:
        return normalize_text(f"сталь {match.group(1)}")

    return text


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
    if "mpa" in text or "мпа" in text:
        return number * 10.0
    return number


def normalize_temperature(value: Any) -> str | None:
    return normalize_text(value)


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


def _extract_number_word(text: str, aliases: dict[str, float]) -> float | None:
    for alias, number in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text):
            return float(number)
    return None


_DN_WORD_ALIASES = {
    "сотка": 100,
    "пятидесятый": 50,
    "пятьдесят": 50,
    "шестнадцать": 16,
    "двадцать пять": 25,
    "сорок": 40,
    "шестьдесят пять": 65,
}

_PN_WORD_ALIASES = {
    "шестнадцать": 16,
    "двадцать пять": 25,
    "двадцать": 20,
    "сорок": 40,
    "пятьдесят": 50,
}

_STANDARD_DN_VALUES = (
    6,
    8,
    10,
    15,
    20,
    25,
    32,
    40,
    50,
    65,
    80,
    100,
    125,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    600,
)

_MAX_SEMANTIC_DN_CORRECTION_DISTANCE = 10


def _nearest_standard_dn(value: float) -> float | None:
    distances = sorted((abs(candidate - value), candidate) for candidate in _STANDARD_DN_VALUES)
    if not distances:
        return None
    best_distance = distances[0][0]
    if best_distance > _MAX_SEMANTIC_DN_CORRECTION_DISTANCE:
        return None
    if len(distances) > 1 and distances[1][0] == best_distance:
        return None
    nearest = distances[0][1]
    return float(nearest)


def _normalize_semantic_dn_candidate(candidate: float) -> float | None:
    rounded = _nearest_standard_dn(candidate)
    if rounded is not None:
        return rounded
    return None


def normalize_semantic_dn(value: Any) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _normalize_semantic_dn_candidate(float(value))

    text = normalize_text(value)
    if text is None:
        return None

    word_value = _extract_number_word(text, _DN_WORD_ALIASES)
    if word_value is not None:
        return word_value

    match = _NUMBER_RE.search(text.replace(",", "."))
    if not match:
        return None

    candidate = float(match.group(0).replace(",", "."))
    return _normalize_semantic_dn_candidate(candidate)


def normalize_semantic_pn_bar(value: Any) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = normalize_text(value)
    if text is None:
        return None

    word_value = _extract_number_word(text, _PN_WORD_ALIASES)
    if word_value is not None:
        return word_value

    number_match = _NUMBER_RE.search(text.replace(",", "."))
    if not number_match:
        return None

    number = float(number_match.group(0).replace(",", "."))
    if "mpa" in text or "мпа" in text:
        return number * 10.0
    return number


__all__ = [
    "ArticleNormalization",
    "normalize_text",
    "normalize_article",
    "normalize_brand",
    "normalize_supported_brand",
    "normalize_connection",
    "normalize_medium",
    "normalize_control",
    "normalize_body_material",
    "normalize_dn",
    "normalize_semantic_dn",
    "normalize_pn_bar",
    "normalize_semantic_pn_bar",
    "normalize_temperature",
    "normalize_length",
]
