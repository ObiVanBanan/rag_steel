from __future__ import annotations

from types import SimpleNamespace

from rag_steel.query_resolver import CompetitorArticleCatalog


def _point(payload: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(payload=payload)


class FakeCatalogClient:
    def __init__(self, points: list[SimpleNamespace]) -> None:
        self.points = points
        self.scroll_calls: list[dict[str, object]] = []

    def get_aliases(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            aliases=[
                SimpleNamespace(
                    alias_name="steel_products_active",
                    collection_name="steel_products_20260818T010203Z",
                )
            ]
        )

    def scroll(self, **kwargs: object) -> tuple[list[SimpleNamespace], None]:
        self.scroll_calls.append(kwargs)
        if len(self.scroll_calls) > 1:
            return [], None
        return self.points, None


def _resolver(points: list[SimpleNamespace]) -> CompetitorArticleCatalog:
    client = FakeCatalogClient(points)
    return CompetitorArticleCatalog(
        client_getter=lambda: client,
        collection_alias="steel_products_active",
    )


def test_resolve_brand_exact_alias_and_fuzzy_typo() -> None:
    resolver = _resolver([])

    exact = resolver.resolve_brand("алсо")
    fuzzy = resolver.resolve_brand("Tempr")
    short = resolver.resolve_brand("ADL")

    assert exact.canonical == "ALSO"
    assert exact.match_type == "exact"
    assert fuzzy.canonical == "Temper"
    assert fuzzy.match_type == "exact"
    assert short.canonical == "ADL"
    assert short.match_type == "exact"


def test_resolve_brand_does_not_guess_ambiguous_or_short_typos() -> None:
    resolver = _resolver([])

    assert resolver.resolve_brand("abc").canonical is None
    assert resolver.resolve_brand("xyz").canonical is None
    assert resolver.resolve_brand("all").canonical is None
    assert resolver.resolve_brand("Valtec").canonical is None


def test_resolve_article_exact_and_fuzzy_unique_match() -> None:
    resolver = _resolver(
        [
            _point(
                {
                    "article": "A0069",
                    "article_norm": "a0069",
                    "article_compact": "a0069",
                    "brand": "Broen",
                    "dn": 50,
                    "pn_bar": 40,
                    "connection": "threaded",
                    "ld_candidates": [],
                }
            ),
            _point(
                {
                    "article": "A0486",
                    "article_norm": "a0486",
                    "article_compact": "a0486",
                    "brand": "Broen",
                    "dn": 50,
                    "pn_bar": 10,
                    "connection": "threaded",
                    "ld_candidates": [],
                }
            ),
        ]
    )

    exact = resolver.resolve_article("A0069")
    fuzzy = resolver.resolve_article("A048б")
    normalized = resolver.resolve_article("a 0069")

    assert exact.article == "A0069"
    assert exact.match_type == "exact"
    assert fuzzy.article == "A0486"
    assert fuzzy.match_type == "fuzzy"
    assert normalized.article == "A0069"
    assert normalized.match_type == "exact"


def test_resolve_article_treats_pn_as_minimum_pressure() -> None:
    resolver = _resolver(
        [
            _point(
                {
                    "article": "A0099",
                    "article_norm": "a0099",
                    "article_compact": "a0099",
                    "brand": "Broen",
                    "dn": 50,
                    "pn_bar": 25,
                    "connection": "threaded",
                    "ld_candidates": [],
                }
            )
        ]
    )

    compatible = resolver.resolve_article("A0099", dn=50, pn_bar=16)
    conflict = resolver.resolve_article("A0099", dn=50, pn_bar=40)

    assert compatible.article == "A0099"
    assert compatible.source_product is not None
    assert compatible.source_product["pn_bar"] == 25
    assert conflict.article is None
    assert conflict.reason_code == "IDENTITY_CONFLICT"


def test_resolve_article_detects_ambiguity_and_conflict() -> None:
    resolver = _resolver(
        [
            _point(
                {
                    "article": "A0486",
                    "article_norm": "a0486",
                    "article_compact": "a0486",
                    "brand": "Broen",
                    "dn": 50,
                    "pn_bar": 10,
                    "connection": "threaded",
                    "ld_candidates": [],
                }
            ),
            _point(
                {
                    "article": "A0488",
                    "article_norm": "a0488",
                    "article_compact": "a0488",
                    "brand": "Broen",
                    "dn": 50,
                    "pn_bar": 10,
                    "connection": "threaded",
                    "ld_candidates": [],
                }
            ),
        ]
    )

    ambiguous = resolver.resolve_article("A0487")
    conflict = resolver.resolve_article("A0486", brand="Temper")

    assert ambiguous.ambiguous is True
    assert ambiguous.reason_code == "ARTICLE_AMBIGUOUS"
    assert conflict.reason_code == "IDENTITY_CONFLICT"
    assert conflict.article is None


def test_resolve_combines_brand_article_and_hard_conflicts() -> None:
    resolver = _resolver(
        [
            _point(
                {
                    "article": "A0069",
                    "article_norm": "a0069",
                    "article_compact": "a0069",
                    "brand": "Broen",
                    "dn": 50,
                    "pn_bar": 40,
                    "connection": "threaded",
                    "ld_candidates": [],
                }
            )
        ]
    )

    ok = resolver.resolve(raw_brand="Broen", raw_article="A0069", dn=50, pn_bar=40)
    hard_conflict = resolver.resolve(raw_brand="Broen", raw_article="A0069", dn=999, pn_bar=40)

    assert ok.reason_code is None
    assert ok.article is not None
    assert ok.article.article == "A0069"
    assert hard_conflict.reason_code == "IDENTITY_CONFLICT"


def test_resolve_article_only_backfills_canonical_brand() -> None:
    resolver = _resolver(
        [
            _point(
                {
                    "article": "A0069",
                    "article_norm": "a0069",
                    "article_compact": "a0069",
                    "brand": "Broen",
                    "dn": 50,
                    "pn_bar": 40,
                    "connection": "threaded",
                    "ld_candidates": [],
                }
            )
        ]
    )

    resolution = resolver.resolve(raw_brand=None, raw_article="A0069")

    assert resolution.resolution_mode == "article_exact"
    assert resolution.brand.canonical == "Broen"
    assert resolution.brand.match_type == "exact"
    assert resolution.article is not None
    assert resolution.article.brand == "Broen"
