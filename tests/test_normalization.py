from __future__ import annotations

import pytest

from rag_steel.normalization import (
    ArticleNormalization,
    normalize_article,
    normalize_brand,
    normalize_connection,
    normalize_control,
    normalize_dn,
    normalize_length,
    normalize_medium,
    normalize_pn_bar,
    normalize_temperature,
    normalize_text,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  Temper\u00a0DN80  PN16  ", "temper dn80 pn16"),
        ("Ёжик", "ежик"),
        ("  Mixed\tCase  Text ", "mixed case text"),
        ("\u212b", "å"),
        (None, None),
        (float("nan"), None),
    ],
)
def test_normalize_text(value, expected):
    assert normalize_text(value) == expected


@pytest.mark.parametrize(
    ("value", "raw", "norm", "compact"),
    [
        ("КШ.П.П.015.40-01", "КШ.П.П.015.40-01", "кш.п.п.015.40-01", "кшпп0154001"),
        ("a-0486", "a-0486", "a-0486", "a0486"),
        (" A / B _ C ", " A / B _ C ", "a / b _ c", "abc"),
        ("Temper 1184399", "Temper 1184399", "temper 1184399", "temper1184399"),
        (None, None, None, None),
    ],
)
def test_normalize_article(value, raw, norm, compact):
    result = normalize_article(value)
    assert result == ArticleNormalization(raw, norm, compact)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Temper", "Temper"),
        ("Кран Broen Ду80", "Broen"),
        ("АЛСО", "ALSO"),
        ("Маршал", "MARSHAL"),
        ("Бивал", "Бивал"),
        ("FORTECA", "FORTECA"),
        ("unknown brand", "unknown brand"),
        (None, None),
    ],
)
def test_normalize_brand(value, expected):
    assert normalize_brand(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Фланцевое", "фланцевое"),
        ("фланцевый кран", "фланцевое"),
        ("threaded", "резьбовое"),
        ("Под приварку", "сварное"),
        ("Муфтовый", "муфтовое"),
        ("без указания", "без указания"),
        (None, None),
    ],
)
def test_normalize_connection(value, expected):
    assert normalize_connection(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Жидкость", "жидкость"),
        ("вода", "жидкость"),
        ("gas", "газ"),
        ("Steam", "пар"),
        ("нефтепродукты", "нефтепродукты"),
        ("среда", "среда"),
        (None, None),
    ],
)
def test_normalize_medium(value, expected):
    assert normalize_medium(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Ручное", "ручное"),
        ("manual", "ручное"),
        ("Пневмопривод", "пневмопривод"),
        ("electric", "электропривод"),
        ("редуктор", "редуктор"),
        ("по месту", "по месту"),
        (None, None),
    ],
)
def test_normalize_control(value, expected):
    assert normalize_control(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Ду80", 80.0),
        ("DN 50", 50.0),
        ("80 мм", 80.0),
        ("Ø100", 100.0),
        (80, 80.0),
        (None, None),
    ],
)
def test_normalize_dn(value, expected):
    assert normalize_dn(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("PN16", 16.0),
        ("Ру25", 25.0),
        ("1,6 МПа", 16.0),
        ("2.5 mpa", 25.0),
        ("4 bar", 4.0),
        (16, 16.0),
        ("неизвестно", None),
        (None, None),
    ],
)
def test_normalize_pn_bar(value, expected):
    assert normalize_pn_bar(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("-40...+80", "-40...+80"),
        ("  от -20 до +120  ", "от -20 до +120"),
        ("COLD", "cold"),
        (None, None),
    ],
)
def test_normalize_temperature(value, expected):
    assert normalize_temperature(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("300", 300.0),
        ("300 мм", 300.0),
        ("30 см", 300.0),
        ("0,3 м", 300.0),
        ("1m", 1000.0),
        (250, 250.0),
        ("неизвестно", None),
        (None, None),
    ],
)
def test_normalize_length(value, expected):
    assert normalize_length(value) == expected
