from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_steel.required_parameters import (
    BRASS_BALL_VALVE,
    BUTTERFLY_VALVE,
    STEEL_BALL_VALVE,
    infer_product_family,
    validate_required_parameters,
)


def _attrs(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "product_family": None,
        "dn": None,
        "pn_bar": None,
        "connection": None,
        "passage_type": None,
        "body_material": None,
        "disc_material": None,
        "seal_type": None,
        "thread_type": None,
        "thread_size": None,
        "medium": None,
        "control": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("family", "values", "expected_required"),
    [
        (
            STEEL_BALL_VALVE,
            {
                "dn": 80,
                "pn_bar": 16,
                "connection": "фланцевое",
                "passage_type": "полнопроходной",
                "medium": "газ",
                "control": "ручное",
                "body_material": "сталь 09г2с",
            },
            (
                "dn",
                "pn",
                "connection",
                "passage_type",
                "medium",
                "control",
                "body_material",
            ),
        ),
        (
            BUTTERFLY_VALVE,
            {
                "dn": 100,
                "pn_bar": 16,
                "body_material": "чугун",
                "disc_material": "нержавеющая сталь",
                "seal_type": "epdm",
                "connection": "межфланцевое",
            },
            (
                "dn",
                "pn",
                "body_material",
                "disc_material",
                "seal_type",
                "connection",
            ),
        ),
        (
            BRASS_BALL_VALVE,
            {
                "dn": 20,
                "pn_bar": 40,
                "thread_type": "внутренняя-внутренняя",
                "thread_size": "3/4",
                "control": "бабочка",
                "medium": "вода",
            },
            ("dn", "pn", "thread_type", "thread_size", "control", "medium"),
        ),
    ],
)
def test_required_parameter_policy_accepts_complete_products(
    family: str,
    values: dict[str, object],
    expected_required: tuple[str, ...],
) -> None:
    check = validate_required_parameters(
        "product",
        _attrs(product_family=family, **values),
    )

    assert check.product_family == family
    assert check.required_fields == expected_required
    assert check.missing_fields == ()
    assert check.complete is True


def test_required_parameter_policy_reports_missing_fields() -> None:
    check = validate_required_parameters(
        "затвор дисковый DN100 PN16",
        _attrs(
            product_family=BUTTERFLY_VALVE,
            dn=100,
            pn_bar=16,
            connection="межфланцевое",
        ),
    )

    assert check.product_family == BUTTERFLY_VALVE
    assert check.missing_fields == ("body_material", "disc_material", "seal_type")
    assert check.complete is False


def test_family_inference_prefers_brass_before_generic_ball_valve() -> None:
    family = infer_product_family(
        "Кран шаровой латунный DN20 PN40 3/4",
        _attrs(body_material="латунь"),
    )

    assert family == BRASS_BALL_VALVE


def test_generic_ball_valve_without_material_is_not_forced_to_steel() -> None:
    family = infer_product_family(
        "Шаровый кран DN50 PN16",
        _attrs(),
    )

    assert family is None
