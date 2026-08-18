from __future__ import annotations

from types import SimpleNamespace

from eval.evaluate_rag_v4 import _hard_violation
from eval.v4_schema import ExpectedAttributes


def test_rag_hard_violation_treats_pn_as_minimum_pressure() -> None:
    expected = ExpectedAttributes(
        resolved_brand="Temper",
        dn=50,
        pn_bar=16,
        connection="flanged",
    )
    higher_pn_response = SimpleNamespace(
        results=[
            SimpleNamespace(
                competitor=SimpleNamespace(
                    brand="Temper",
                    dn=50,
                    pn_bar=25,
                    connection="flanged",
                )
            )
        ]
    )
    lower_pn_response = SimpleNamespace(
        results=[
            SimpleNamespace(
                competitor=SimpleNamespace(
                    brand="Temper",
                    dn=50,
                    pn_bar=10,
                    connection="flanged",
                )
            )
        ]
    )

    assert _hard_violation(expected, higher_pn_response) is False
    assert _hard_violation(expected, lower_pn_response) is True
