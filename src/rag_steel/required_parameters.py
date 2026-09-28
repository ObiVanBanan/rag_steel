"""Required-parameter policy for supported valve product families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_steel.normalization import normalize_text

STEEL_BALL_VALVE = "steel_ball_valve"
BUTTERFLY_VALVE = "butterfly_valve"
BRASS_BALL_VALVE = "brass_ball_valve"

_REQUIRED_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    STEEL_BALL_VALVE: (
        ("dn", "dn"),
        ("pn", "pn_bar"),
        ("connection", "connection"),
        ("passage_type", "passage_type"),
        ("medium", "medium"),
        ("control", "control"),
        ("body_material", "body_material"),
    ),
    BUTTERFLY_VALVE: (
        ("dn", "dn"),
        ("pn", "pn_bar"),
        ("body_material", "body_material"),
        ("disc_material", "disc_material"),
        ("seal_type", "seal_type"),
        ("connection", "connection"),
    ),
    BRASS_BALL_VALVE: (
        ("dn", "dn"),
        ("pn", "pn_bar"),
        ("thread_type", "thread_type"),
        ("thread_size", "thread_size"),
        ("control", "control"),
        ("medium", "medium"),
    ),
}

_FAMILY_ALIASES = {
    "steel_ball_valve": STEEL_BALL_VALVE,
    "steel valve": STEEL_BALL_VALVE,
    "steel ball valve": STEEL_BALL_VALVE,
    "стальной кран": STEEL_BALL_VALVE,
    "стальной шаровой кран": STEEL_BALL_VALVE,
    "butterfly_valve": BUTTERFLY_VALVE,
    "butterfly valve": BUTTERFLY_VALVE,
    "дисковый затвор": BUTTERFLY_VALVE,
    "затвор дисковый": BUTTERFLY_VALVE,
    "brass_ball_valve": BRASS_BALL_VALVE,
    "brass valve": BRASS_BALL_VALVE,
    "brass ball valve": BRASS_BALL_VALVE,
    "латунный кран": BRASS_BALL_VALVE,
    "латунный шаровой кран": BRASS_BALL_VALVE,
}


@dataclass(frozen=True, slots=True)
class RequiredParameterCheck:
    product_family: str | None
    required_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_fields


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _explicit_family(value: Any) -> str | None:
    normalized = normalize_text(value)
    if normalized is None:
        return None
    return _FAMILY_ALIASES.get(normalized)


def infer_product_family(query: str, attributes: Any) -> str | None:
    """Infer only the three product families with an explicit required-field policy."""

    explicit = _explicit_family(getattr(attributes, "product_family", None))
    if explicit is not None:
        return explicit

    query_text = normalize_text(query) or ""
    body_material = normalize_text(getattr(attributes, "body_material", None)) or ""

    if (
        "дисков" in query_text
        or "butterfly" in query_text
        or ("затвор" in query_text and "диск" in query_text)
    ):
        return BUTTERFLY_VALVE

    if "латун" in query_text or "brass" in query_text or "латун" in body_material:
        return BRASS_BALL_VALVE

    is_ball_valve = (
        ("кран" in query_text and "шар" in query_text)
        or "ball valve" in query_text
    )
    is_steel = (
        "сталь" in query_text
        or "сталь" in body_material
        or "steel" in query_text
        or "steel" in body_material
    )
    if is_ball_valve and is_steel:
        return STEEL_BALL_VALVE

    return None


def validate_required_parameters(query: str, attributes: Any) -> RequiredParameterCheck:
    family = infer_product_family(query, attributes)
    if family is None:
        return RequiredParameterCheck(
            product_family=None,
            required_fields=(),
            missing_fields=(),
        )

    policy = _REQUIRED_FIELDS[family]
    missing = tuple(
        public_name
        for public_name, attribute_name in policy
        if not _has_value(getattr(attributes, attribute_name, None))
    )
    return RequiredParameterCheck(
        product_family=family,
        required_fields=tuple(public_name for public_name, _ in policy),
        missing_fields=missing,
    )


__all__ = [
    "BRASS_BALL_VALVE",
    "BUTTERFLY_VALVE",
    "STEEL_BALL_VALVE",
    "RequiredParameterCheck",
    "infer_product_family",
    "validate_required_parameters",
]
