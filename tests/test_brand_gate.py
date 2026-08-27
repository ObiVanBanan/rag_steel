from __future__ import annotations

from rag_steel.brand_gate import detect_competitor_brand


def test_detect_competitor_brand_uses_token_boundaries() -> None:
    assert detect_competitor_brand("бронзовый кран") is None
    assert detect_competitor_brand("badly") is None


def test_detect_competitor_brand_requires_product_context_for_ambiguous_aliases() -> None:
    assert detect_competitor_brand("we also need") is None
    assert detect_competitor_brand("ALSO DN80 PN16") == "ALSO"


def test_detect_competitor_brand_still_matches_supported_brands() -> None:
    assert detect_competitor_brand("Temper DN80 PN16") == "Temper"
    assert detect_competitor_brand("ADL DN80 PN16") == "ADL"
    assert detect_competitor_brand("Брон DN80 PN16") == "Broen"


def test_detect_competitor_brand_treats_gate_valves_as_product_context() -> None:
    assert detect_competitor_brand("ALSO затвор фланцевый") == "ALSO"
