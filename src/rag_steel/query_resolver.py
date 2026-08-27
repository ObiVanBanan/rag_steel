"""Deterministic brand and article resolution for query interpretation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_steel.competitor_registry import COMPETITOR_BRANDS
from rag_steel.normalization import (
    normalize_article,
    normalize_body_material,
    normalize_brand,
    normalize_connection,
    normalize_supported_brand,
    normalize_text,
)

_BRAND_CANDIDATES: dict[str, tuple[str, ...]] = {
    canonical: tuple(
        dict.fromkeys(
            token
            for token in (
                normalize_text(canonical),
                *(normalize_text(alias) for alias in aliases),
            )
            if token and len(token) >= 4
        )
    )
    for canonical, aliases in COMPETITOR_BRANDS.items()
}


def _normalize_raw_fragment(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\xa0", " ").strip()
    text = " ".join(text.split())
    return text or None


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


def _normalize_article_key(value: Any) -> str | None:
    normalized = normalize_article(value)
    if normalized.article_compact:
        return normalized.article_compact.casefold()
    if normalized.article_norm:
        return normalized.article_norm.casefold()
    if normalized.article_raw is not None:
        return _normalize_raw_fragment(normalized.article_raw)
    return None


def _pn_meets_minimum(candidate_pn: Any, requested_pn: Any) -> bool:
    try:
        return float(candidate_pn) >= float(requested_pn)
    except (TypeError, ValueError):
        return False


@dataclass(slots=True)
class ResolvedBrand:
    raw: str | None
    canonical: str | None
    match_type: str | None
    distance: int | None = None
    reason_code: str | None = None


@dataclass(slots=True)
class ResolvedArticle:
    raw: str | None
    normalized: str | None
    compact: str | None
    article: str | None
    brand: str | None
    match_type: str | None
    distance: int | None = None
    ambiguous: bool = False
    reason_code: str | None = None
    source_product: dict[str, Any] | None = None
    exact_candidates: int = 0
    logical_candidates: int = 0


@dataclass(slots=True)
class QueryResolution:
    raw_brand: str | None
    raw_article: str | None
    brand: ResolvedBrand
    article: ResolvedArticle | None
    resolution_mode: str
    reason_code: str | None = None
    retryable: bool = False


@dataclass(slots=True)
class CompetitorArticleCatalog:
    client_getter: Any
    collection_alias: str
    _cached_collection_name: str | None = field(init=False, default=None, repr=False)
    _cached_records: list[dict[str, Any]] = field(init=False, default_factory=list, repr=False)

    @staticmethod
    def _resolve_physical_collection_name(client: Any, collection_name: str) -> str:
        if not collection_name:
            return collection_name
        try:
            aliases = client.get_aliases()
        except Exception:
            return collection_name
        for alias in getattr(aliases, "aliases", []) or []:
            if getattr(alias, "alias_name", None) == collection_name:
                resolved = getattr(alias, "collection_name", None)
                if resolved:
                    return str(resolved)
        return collection_name

    @staticmethod
    def _extract_payload(point: Any) -> dict[str, Any]:
        payload = getattr(point, "payload", None)
        if payload is None and isinstance(point, dict):
            payload = point.get("payload")
        return dict(payload or {})

    @staticmethod
    def _payload_text(payload: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            text = _normalize_raw_fragment(value)
            if text:
                return text
        return None

    @staticmethod
    def _payload_number(payload: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _normalize_record(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        article = self._payload_text(payload, "article", "steel_article", "ld_article")
        if article is None:
            return None

        article_normalization = normalize_article(article)
        brand = normalize_brand(
            self._payload_text(payload, "brand", "steel_brand", "name", "steel_name")
        )
        if brand is None:
            name = self._payload_text(payload, "name", "steel_name")
            brand = normalize_brand(name)

        ld_candidates = payload.get("ld_candidates") or []
        normalized_ld_candidates: list[dict[str, Any]] = []
        for candidate in ld_candidates:
            if hasattr(candidate, "model_dump"):
                candidate = candidate.model_dump(mode="json")
            if isinstance(candidate, dict):
                normalized_ld_candidates.append(dict(candidate))

        return {
            "article": article,
            "article_norm": self._payload_text(payload, "article_norm", "steel_article_norm")
            or article_normalization.article_norm
            or article.casefold(),
            "article_compact": self._payload_text(
                payload,
                "article_compact",
                "steel_article_compact",
            )
            or article_normalization.article_compact
            or article_normalization.article_norm
            or article.casefold(),
            "name": self._payload_text(payload, "name", "steel_name"),
            "brand": brand,
            "dn": self._payload_number(payload, "dn", "steel_dn"),
            "pn_bar": self._payload_number(payload, "pn_bar", "steel_pn_bar"),
            "connection": normalize_connection(
                self._payload_text(payload, "connection", "steel_connection")
            ),
            "body_material": normalize_body_material(
                self._payload_text(payload, "body_material", "steel_body_material")
            ),
            "medium": self._payload_text(payload, "medium", "steel_medium"),
            "control": self._payload_text(payload, "control", "steel_control"),
            "temperature": self._payload_text(payload, "temperature", "steel_temp"),
            "length_mm": self._payload_number(payload, "length_mm", "steel_length"),
            "url": self._payload_text(payload, "url", "steel_url"),
            "ld_candidates": normalized_ld_candidates,
        }

    def _scroll_records(self, client: Any, collection_name: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        offset: Any = None

        while True:
            response = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if isinstance(response, tuple):
                points, offset = response
            else:
                points = (
                    getattr(response, "points", None) or getattr(response, "result", None) or []
                )
                offset = getattr(response, "next_page_offset", None) or getattr(
                    response,
                    "next_offset",
                    None,
                )
            if not points:
                break

            for point in points:
                payload = self._extract_payload(point)
                record = self._normalize_record(payload)
                if record is not None:
                    records.append(record)

            if not offset:
                break

        return records

    def _load_records(self) -> list[dict[str, Any]]:
        client = self.client_getter()
        collection_name = self._resolve_physical_collection_name(client, self.collection_alias)
        if self._cached_collection_name == collection_name and self._cached_records:
            return self._cached_records

        records = self._scroll_records(client, collection_name)
        self._cached_collection_name = collection_name
        self._cached_records = records
        return records

    @staticmethod
    def _record_article_keys(record: dict[str, Any]) -> set[str]:
        keys: set[str] = set()
        for key in ("article", "article_norm", "article_compact"):
            value = record.get(key)
            if value:
                keys.add(_normalize_article_key(value) or str(value).casefold())
                keys.add(" ".join(str(value).split()).casefold())
        return {key for key in keys if key}

    @staticmethod
    def _record_matches_brand(record: dict[str, Any], brand: str | None) -> bool:
        if brand is None:
            return True
        return normalize_brand(record.get("brand") or record.get("name")) == normalize_brand(brand)

    @staticmethod
    def _record_matches_hard_constraints(
        record: dict[str, Any],
        *,
        dn: float | None = None,
        pn_bar: float | None = None,
        connection: str | None = None,
    ) -> bool:
        if dn is not None and record.get("dn") is not None and float(record["dn"]) != float(dn):
            return False
        if pn_bar is not None and not _pn_meets_minimum(record.get("pn_bar"), pn_bar):
            return False
        if connection is not None and normalize_connection(
            record.get("connection")
        ) != normalize_connection(connection):
            return False
        return True

    @staticmethod
    def _record_identity_key(record: dict[str, Any]) -> tuple[str, str]:
        article = _normalize_article_key(
            record.get("article_compact") or record.get("article_norm") or record.get("article")
        ) or ""
        brand = normalize_brand(record.get("brand") or record.get("name")) or ""
        return article, brand

    @staticmethod
    def _merge_records(records: list[dict[str, Any]]) -> dict[str, Any]:
        canonical = dict(records[0])
        merged_ld_candidates: list[dict[str, Any]] = []
        seen_ld_keys: set[str] = set()

        for record in records:
            for key, value in record.items():
                if key == "ld_candidates":
                    continue
                if canonical.get(key) is None and value is not None:
                    canonical[key] = value
            for candidate in record.get("ld_candidates") or []:
                if not isinstance(candidate, dict):
                    continue
                candidate_article = _normalize_article_key(
                    candidate.get("article") or candidate.get("ld_article")
                ) or ""
                candidate_url = _normalize_raw_fragment(candidate.get("url")) or ""
                dedup_key = f"{candidate_article}|{candidate_url}"
                if dedup_key in seen_ld_keys:
                    continue
                seen_ld_keys.add(dedup_key)
                merged_ld_candidates.append(dict(candidate))

        canonical["ld_candidates"] = merged_ld_candidates
        return canonical

    def _deduplicate_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(self._record_identity_key(record), []).append(record)
        return [self._merge_records(group) for group in grouped.values()]

    def _score_article_candidate(self, query_key: str, record: dict[str, Any]) -> int | None:
        candidate_key = (
            record.get("article_compact") or record.get("article_norm") or record.get("article")
        )
        if candidate_key is None:
            return None
        candidate = _normalize_article_key(candidate_key)
        if candidate is None:
            return None
        return _damerau_distance_at_most_one(query_key, candidate)

    def resolve_brand(self, raw_brand: Any) -> ResolvedBrand:
        raw = _normalize_raw_fragment(raw_brand)
        if raw is None:
            return ResolvedBrand(raw=None, canonical=None, match_type=None)

        normalized = normalize_text(raw)
        if normalized is None:
            return ResolvedBrand(raw=raw, canonical=None, match_type=None)

        exact = normalize_supported_brand(normalized)
        if exact is not None:
            return ResolvedBrand(raw=raw, canonical=exact, match_type="exact")

        if len(normalized) < 4:
            return ResolvedBrand(
                raw=raw,
                canonical=None,
                match_type=None,
                reason_code="BRAND_TOO_SHORT",
            )

        best_distance: int | None = None
        best_candidates: list[str] = []
        for canonical, aliases in _BRAND_CANDIDATES.items():
            candidate_distance: int | None = None
            for alias in aliases:
                distance = _damerau_distance_at_most_one(normalized, alias)
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
            return ResolvedBrand(
                raw=raw,
                canonical=None,
                match_type=None,
                reason_code="BRAND_AMBIGUOUS",
            )

        return ResolvedBrand(
            raw=raw,
            canonical=best_candidates[0],
            match_type="fuzzy",
            distance=best_distance,
        )

    def resolve_article(
        self,
        raw_article: Any,
        *,
        brand: str | None = None,
        dn: float | None = None,
        pn_bar: float | None = None,
        connection: str | None = None,
    ) -> ResolvedArticle:
        raw = _normalize_raw_fragment(raw_article)
        if raw is None:
            return ResolvedArticle(
                raw=None,
                normalized=None,
                compact=None,
                article=None,
                brand=brand,
                match_type=None,
            )

        normalized = normalize_article(raw)
        normalized_key = (normalized.article_norm or raw.casefold()).casefold()
        compact_key = (
            normalized.article_compact or normalized.article_norm or raw.casefold()
        ).casefold()
        records = self._load_records()

        exact_candidates = [
            record
            for record in records
            if normalized_key in self._record_article_keys(record)
            or compact_key in self._record_article_keys(record)
        ]
        if exact_candidates:
            logical_candidates = self._deduplicate_records(exact_candidates)
            matched = [
                record
                for record in logical_candidates
                if self._record_matches_brand(record, brand)
                and self._record_matches_hard_constraints(
                    record, dn=dn, pn_bar=pn_bar, connection=connection
                )
            ]
            if len(matched) == 1:
                source_product = dict(matched[0])
                return ResolvedArticle(
                    raw=raw,
                    normalized=normalized.article_norm,
                    compact=normalized.article_compact,
                    article=source_product.get("article"),
                    brand=source_product.get("brand") or brand,
                    match_type="exact",
                    distance=0,
                    source_product=source_product,
                    exact_candidates=len(exact_candidates),
                    logical_candidates=len(logical_candidates),
                )
            if not matched:
                return ResolvedArticle(
                    raw=raw,
                    normalized=normalized.article_norm,
                    compact=normalized.article_compact,
                    article=None,
                    brand=brand,
                    match_type=None,
                    ambiguous=False,
                    reason_code="IDENTITY_CONFLICT",
                    exact_candidates=len(exact_candidates),
                    logical_candidates=len(logical_candidates),
                )
            return ResolvedArticle(
                raw=raw,
                normalized=normalized.article_norm,
                compact=normalized.article_compact,
                article=None,
                brand=brand,
                match_type=None,
                ambiguous=True,
                reason_code="ARTICLE_AMBIGUOUS",
                exact_candidates=len(exact_candidates),
                logical_candidates=len(logical_candidates),
            )

        query_key = compact_key or normalized_key
        if len(query_key) < 5:
            return ResolvedArticle(
                raw=raw,
                normalized=normalized.article_norm,
                compact=normalized.article_compact,
                article=None,
                brand=brand,
                match_type=None,
                reason_code="ARTICLE_TOO_SHORT",
            )

        logical_records = self._deduplicate_records(records)
        scored: list[tuple[int, dict[str, Any]]] = []
        all_scored: list[tuple[int, dict[str, Any]]] = []
        for record in logical_records:
            distance = self._score_article_candidate(query_key, record)
            if distance is None or distance > 1:
                continue
            all_scored.append((distance, record))
            if not self._record_matches_brand(record, brand):
                continue
            if not self._record_matches_hard_constraints(
                record, dn=dn, pn_bar=pn_bar, connection=connection
            ):
                continue
            scored.append((distance, record))

        if not scored:
            if all_scored:
                return ResolvedArticle(
                    raw=raw,
                    normalized=normalized.article_norm,
                    compact=normalized.article_compact,
                    article=None,
                    brand=brand,
                    match_type=None,
                    reason_code="IDENTITY_CONFLICT",
                    logical_candidates=len(all_scored),
                )
            return ResolvedArticle(
                raw=raw,
                normalized=normalized.article_norm,
                compact=normalized.article_compact,
                article=None,
                    brand=brand,
                    match_type=None,
                    reason_code="ARTICLE_NOT_FOUND",
                    logical_candidates=0,
                )

        best_distance = min(distance for distance, _ in scored)
        best_records = [record for distance, record in scored if distance == best_distance]
        if len(best_records) != 1:
            return ResolvedArticle(
                raw=raw,
                normalized=normalized.article_norm,
                compact=normalized.article_compact,
                article=None,
                brand=brand,
                match_type=None,
                distance=best_distance,
                ambiguous=True,
                reason_code="ARTICLE_AMBIGUOUS",
                logical_candidates=len(best_records),
            )

        source_product = dict(best_records[0])
        return ResolvedArticle(
            raw=raw,
            normalized=normalized.article_norm,
            compact=normalized.article_compact,
            article=source_product.get("article"),
            brand=source_product.get("brand") or brand,
            match_type="fuzzy" if best_distance else "exact",
            distance=best_distance,
            source_product=source_product,
            logical_candidates=len(best_records),
        )

    def resolve(
        self,
        *,
        brand: Any | None = None,
        article: Any | None = None,
        raw_brand: Any | None = None,
        raw_article: Any | None = None,
        dn: float | None = None,
        pn_bar: float | None = None,
        connection: str | None = None,
    ) -> QueryResolution:
        if raw_brand is None:
            raw_brand = brand
        if raw_article is None:
            raw_article = article

        brand = self.resolve_brand(raw_brand)
        article = self.resolve_article(
            raw_article,
            brand=brand.canonical if brand.canonical is not None else None,
            dn=dn,
            pn_bar=pn_bar,
            connection=connection,
        )

        if brand.canonical is None and brand.raw is not None:
            return QueryResolution(
                raw_brand=brand.raw,
                raw_article=article.raw,
                brand=brand,
                article=article,
                resolution_mode="brand_unknown",
                reason_code=brand.reason_code or "UNSUPPORTED_COMPETITOR_BRAND",
            )

        if article.reason_code == "ARTICLE_AMBIGUOUS":
            return QueryResolution(
                raw_brand=brand.raw,
                raw_article=article.raw,
                brand=brand,
                article=article,
                resolution_mode="article_ambiguous",
                reason_code="ARTICLE_AMBIGUOUS",
            )

        if article.reason_code in {"ARTICLE_NOT_FOUND", "IDENTITY_CONFLICT"}:
            return QueryResolution(
                raw_brand=brand.raw,
                raw_article=article.raw,
                brand=brand,
                article=article,
                resolution_mode=(
                    "identity_conflict"
                    if article.reason_code == "IDENTITY_CONFLICT"
                    else "article_not_found"
                ),
                reason_code=article.reason_code,
            )

        if article.article is not None and brand.canonical is not None:
            article_brand = article.brand
            if article_brand is not None and normalize_brand(article_brand) != normalize_brand(
                brand.canonical
            ):
                return QueryResolution(
                    raw_brand=brand.raw,
                    raw_article=article.raw,
                    brand=brand,
                    article=ResolvedArticle(
                        raw=article.raw,
                        normalized=article.normalized,
                        compact=article.compact,
                        article=None,
                        brand=article.brand,
                        match_type=article.match_type,
                        distance=article.distance,
                        ambiguous=True,
                        reason_code="IDENTITY_CONFLICT",
                        source_product=article.source_product,
                    ),
                    resolution_mode="identity_conflict",
                    reason_code="IDENTITY_CONFLICT",
                )

        if brand.canonical is None and article.article is None:
            return QueryResolution(
                raw_brand=brand.raw,
                raw_article=article.raw,
                brand=brand,
                article=article,
                resolution_mode="no_identity",
                reason_code="COMPETITOR_BRAND_REQUIRED",
            )

        if article.article is not None and brand.canonical is None:
            resolved_brand = ResolvedBrand(
                raw=brand.raw,
                canonical=article.brand,
                match_type=article.match_type,
                distance=article.distance,
            )
            return QueryResolution(
                raw_brand=brand.raw,
                raw_article=article.raw,
                brand=resolved_brand,
                article=article,
                resolution_mode=f"article_{article.match_type or 'resolved'}",
            )

        if brand.canonical is not None and article.article is None:
            return QueryResolution(
                raw_brand=brand.raw,
                raw_article=article.raw,
                brand=brand,
                article=article,
                resolution_mode=f"brand_{brand.match_type or 'resolved'}",
            )

        return QueryResolution(
            raw_brand=brand.raw,
            raw_article=article.raw,
            brand=brand,
            article=article,
            resolution_mode="brand_and_article",
        )


__all__ = [
    "CompetitorArticleCatalog",
    "QueryResolution",
    "ResolvedArticle",
    "ResolvedBrand",
]
