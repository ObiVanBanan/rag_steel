"""Canonical competitor brand registry."""

from __future__ import annotations

COMPETITOR_BRANDS: dict[str, tuple[str, ...]] = {
    "Temper": ("temper", "темпер", "темпр"),
    "ALSO": ("also", "алсо"),
    "MARSHAL": ("marshal", "маршал"),
    "Broen": ("broen", "брон", "броен"),
    "ADL": ("adl", "адл"),
    "FORTECA": ("forteca", "фортека"),
    "Бивал": ("бивал", "bival"),
    "PALUR": ("palur", "палур"),
    "Gallop": ("gallop", "галлоп", "галоп"),
    "Aquasfera": ("aquasfera", "аквасфера"),
    "БАЗ": ("баз", "baz"),
    "Stout": ("stout", "стаут"),
    "STI": ("sti", "сти"),
    "Valtec": ("valtec", "валтек"),
}


def iter_brand_aliases() -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    for canonical, brand_aliases in COMPETITOR_BRANDS.items():
        aliases.append((canonical, canonical))
        for alias in brand_aliases:
            aliases.append((alias, canonical))
    return aliases


__all__ = ["COMPETITOR_BRANDS", "iter_brand_aliases"]
