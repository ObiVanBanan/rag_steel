from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from rag_steel.indexer import build_index
from rag_steel.schemas import SteelProductDocument


@dataclass(slots=True)
class FakeModel:
    calls: list[dict[str, object]]

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
            }
        )
        return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.created_collections: list[dict[str, object]] = []
        self.upserts: list[dict[str, object]] = []
        self.query_calls: list[dict[str, object]] = []
        self.alias_operations: list[list[object]] = []
        self.sample_payload = {
            "steel_id": "doc-1",
            "article": "КШ.П.П.015.40-01",
            "semantic_text": "SOURCE_SENTINEL",
            "lexical_text": "SOURCE_SENTINEL",
            "ld_candidates": [],
        }

    def collection_exists(self, collection_name: str, **_: object) -> bool:
        return False

    def create_collection(self, **kwargs: object) -> bool:
        self.created_collections.append(kwargs)
        return True

    def upsert(
        self,
        *,
        collection_name: str,
        points: list[object],
        wait: bool = True,
        **_: object,
    ) -> object:
        self.upserts.append(
            {"collection_name": collection_name, "points": points, "wait": wait}
        )
        return SimpleNamespace()

    def count(self, *, collection_name: str, exact: bool = True, **_: object) -> object:
        return SimpleNamespace(count=sum(len(item["points"]) for item in self.upserts))

    def scroll(
        self,
        *,
        collection_name: str,
        limit: int = 1,
        **_: object,
    ) -> tuple[list[object], object]:
        return [SimpleNamespace(payload=self.sample_payload)], None

    def query_points(self, **kwargs: object) -> object:
        self.query_calls.append(kwargs)
        return SimpleNamespace(points=[SimpleNamespace(id="hit-1")])

    def get_aliases(self, **_: object) -> object:
        return SimpleNamespace(
            aliases=[SimpleNamespace(alias_name="steel_products_active", collection_name="old")]
        )

    def update_collection_aliases(self, operations: list[object], **_: object) -> bool:
        self.alias_operations.append(operations)
        return True


def _make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ld_name": "LD",
                "ld_article": "LD-1",
                "ld_url": "https://ld.example/1",
                "ld_dn": 80,
                "ld_pn_mpa": 1.6,
                "ld_connection": "фланцевое",
                "ld_medium": "жидкость",
                "ld_control": "ручное",
                "ld_temp": None,
                "ld_length": 300,
                "steel_name": "SOURCE_SENTINEL",
                "steel_article": "КШ.П.П.015.40-01",
                "steel_url": "https://steel.example/1",
                "steel_dn": 80,
                "steel_pn_bar": 16,
                "steel_connection": "фланцевый",
                "steel_medium": "жидкость",
                "steel_control": "ручное",
                "steel_temp": "до +80",
                "steel_length": "300 мм",
                "match_score": 8,
                "match_max": 7,
                "price_ld": 12130,
            },
            {
                "ld_name": "LD",
                "ld_article": "LD-2",
                "ld_url": "https://ld.example/2",
                "ld_dn": 50,
                "ld_pn_mpa": 1.6,
                "ld_connection": "фланцевое",
                "ld_medium": "жидкость",
                "ld_control": "ручное",
                "ld_temp": None,
                "ld_length": 250,
                "steel_name": "SOURCE_SENTINEL",
                "steel_article": "а0486",
                "steel_url": "https://steel.example/2",
                "steel_dn": 50,
                "steel_pn_bar": 16,
                "steel_connection": "фланцевый",
                "steel_medium": "жидкость",
                "steel_control": "ручное",
                "steel_temp": "до +80",
                "steel_length": "250 мм",
                "match_score": 7,
                "match_max": 7,
                "price_ld": 9800,
            },
        ]
    )


def _make_documents() -> list[SteelProductDocument]:
    return [
        SteelProductDocument(
            steel_id="doc-1",
            article="КШ.П.П.015.40-01",
            article_norm="кш.п.п.015.40-01",
            article_compact="кшпп0154001",
            name="SOURCE_SENTINEL",
            name_variants=["SOURCE_SENTINEL"],
            brand="Temper",
            dn=80.0,
            pn_bar=16.0,
            connection="фланцевое",
            medium="жидкость",
            control="ручное",
            temperature="до +80",
            length_mm=300.0,
            url="https://steel.example/1",
            semantic_text="SOURCE_SENTINEL semantic",
            lexical_text="SOURCE_SENTINEL lexical",
        ),
        SteelProductDocument(
            steel_id="doc-2",
            article="а0486",
            article_norm="а0486",
            article_compact="а0486",
            name="SOURCE_SENTINEL",
            name_variants=["SOURCE_SENTINEL"],
            brand="Temper",
            dn=50.0,
            pn_bar=16.0,
            connection="фланцевое",
            medium="жидкость",
            control="ручное",
            temperature="до +80",
            length_mm=250.0,
            url="https://steel.example/2",
            semantic_text="SOURCE_SENTINEL semantic 2",
            lexical_text="SOURCE_SENTINEL lexical 2",
        ),
    ]


def test_build_index_batches_embeddings_and_switches_alias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "mapping_results.csv"
    _make_frame().to_csv(csv_path, index=False)
    metadata_path = tmp_path / "index_build.json"
    fake_client = FakeQdrantClient()
    fake_model = FakeModel(calls=[])
    documents = _make_documents()

    monkeypatch.setattr(
        "rag_steel.indexer.build_source_documents_from_frame",
        lambda df: documents,
    )

    result = build_index(
        csv_path,
        model_name="test-model",
        recreate=True,
        client=fake_client,
        metadata_path=metadata_path,
        batch_size=2,
        build_time=datetime(2026, 8, 4, 12, 34, 56, tzinfo=timezone.utc),
        smoke_queries=[
            "1184399",
            "а0486",
            "Temper DN80 PN16",
            "Broen Ду80 Ру16",
            "фланцевый кран Ду50 Ру40",
        ],
        model_factory=lambda: fake_model,
    )

    assert result.metadata.collection_name == "steel_products_test-model_20260804T123456Z"
    assert result.metadata_path == metadata_path
    assert result.metadata.document_count == 2
    assert result.metadata.source_row_count == 2
    assert result.metadata.deduplicated_row_count == 2
    assert metadata_path.exists()
    assert "csv_sha256" in metadata_path.read_text(encoding="utf-8")

    assert len(fake_model.calls) == 2
    assert fake_model.calls[0]["texts"] == [document.semantic_text for document in documents]
    assert fake_model.calls[0]["batch_size"] == 2
    assert fake_model.calls[0]["normalize_embeddings"] is True
    assert fake_model.calls[0]["show_progress_bar"] is True
    assert fake_model.calls[1]["texts"] == [
        "1184399",
        "а0486",
        "Temper DN80 PN16",
        "Broen Ду80 Ру16",
        "фланцевый кран Ду50 Ру40",
    ]
    assert fake_model.calls[1]["show_progress_bar"] is False

    assert len(fake_client.created_collections) == 1
    assert fake_client.created_collections[0]["collection_name"] == result.metadata.collection_name
    assert len(fake_client.upserts) == 1
    assert len(fake_client.upserts[0]["points"]) == 2
    assert (
        fake_client.upserts[0]["points"][0].payload["semantic_text"]
        == "SOURCE_SENTINEL semantic"
    )
    assert fake_client.upserts[0]["points"][0].vector["sparse"].text == "SOURCE_SENTINEL lexical"
    assert len(fake_client.query_calls) == 5
    assert len(fake_client.alias_operations) == 1
    assert len(fake_client.alias_operations[0]) == 2


def test_build_index_skips_alias_switch_without_recreate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "mapping_results.csv"
    _make_frame().to_csv(csv_path, index=False)
    fake_client = FakeQdrantClient()
    fake_model = FakeModel(calls=[])

    monkeypatch.setattr(
        "rag_steel.indexer.build_source_documents_from_frame",
        lambda df: _make_documents(),
    )

    build_index(
        csv_path,
        model_name="test-model",
        recreate=False,
        client=fake_client,
        metadata_path=tmp_path / "index_build.json",
        batch_size=2,
        build_time=datetime(2026, 8, 4, 12, 34, 56, tzinfo=timezone.utc),
        smoke_queries=[
            "1184399",
            "а0486",
            "Temper DN80 PN16",
            "Broen Ду80 Ру16",
            "фланцевый кран Ду50 Ру40",
        ],
        model_factory=lambda: fake_model,
    )

    assert fake_client.alias_operations == []
