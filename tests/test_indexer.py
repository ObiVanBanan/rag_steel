from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pandas as pd
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_steel.indexer import (
    _point_id_for,
    _upsert_with_retry,
    _wait_for_qdrant_ready,
    build_index,
)
from rag_steel.schemas import SteelProductDocument
from rag_steel.source_adapters import SourceFileRecord


@dataclass(slots=True)
class FakeEmbedder:
    calls: list[dict[str, object]]
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    dimension: int = 384
    embedding_revision: str = ""
    embedding_dtype: str = "float32"
    max_sequence_length: int = 512

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

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
        return [
            [float(index + 1), *([0.0] * (self.dimension - 1))] for index, _ in enumerate(texts)
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encode(
            texts,
            batch_size=max(1, len(texts)),
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def embed_query(self, text: str) -> list[float]:
        return self.encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.created_collections: list[dict[str, object]] = []
        self.update_collection_calls: list[dict[str, object]] = []
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

    def update_collection(self, **kwargs: object) -> bool:
        self.update_collection_calls.append(kwargs)
        return True

    def upsert(
        self,
        *,
        collection_name: str,
        points: list[object],
        wait: bool = True,
        **_: object,
    ) -> object:
        self.upserts.append({"collection_name": collection_name, "points": points, "wait": wait})
        return SimpleNamespace()

    def count(self, *, collection_name: str, exact: bool = True, **_: object) -> object:
        return SimpleNamespace(count=sum(len(item["points"]) for item in self.upserts))

    def scroll(
        self, *, collection_name: str, limit: int = 1, **_: object
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


class FlakyReadyQdrantClient(FakeQdrantClient):
    def __init__(self, failures_before_ready: int) -> None:
        super().__init__()
        self.failures_before_ready = failures_before_ready
        self.collection_exists_calls = 0

    def collection_exists(self, collection_name: str, **_: object) -> bool:
        self.collection_exists_calls += 1
        if self.collection_exists_calls <= self.failures_before_ready:
            raise UnexpectedResponse(
                status_code=503,
                reason_phrase="Service Unavailable",
                content=b"",
                headers={},
            )
        return False


class FlakyUpsertQdrantClient(FakeQdrantClient):
    def __init__(self, failures_before_success: int) -> None:
        super().__init__()
        self.failures_before_success = failures_before_success
        self.upsert_attempts = 0

    def upsert(
        self,
        *,
        collection_name: str,
        points: list[object],
        wait: bool = True,
        **_: object,
    ) -> object:
        self.upsert_attempts += 1
        if self.upsert_attempts <= self.failures_before_success:
            raise RuntimeError("timed out")
        return super().upsert(collection_name=collection_name, points=points, wait=wait)


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


def _make_source_files() -> list[SourceFileRecord]:
    return [
        SourceFileRecord(
            name="mapping_results.csv",
            adapter="canonical",
            sha256="a" * 64,
            rows=2,
        )
    ]


def test_build_index_batches_embeddings_and_switches_alias(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "mapping_results.csv"
    _make_frame().to_csv(csv_path, index=False)
    metadata_path = tmp_path / "index_build.json"
    fake_client = FakeQdrantClient()
    fake_model = FakeEmbedder(calls=[])
    documents = _make_documents()
    source_rows = _make_frame()
    source_files = _make_source_files()

    monkeypatch.setattr(
        "rag_steel.indexer.load_source_bundle",
        lambda paths: (source_rows, source_files),
    )
    monkeypatch.setattr(
        "rag_steel.indexer.build_source_documents_from_frame",
        lambda df: documents,
    )

    result = build_index(
        csv_path,
        embedder=fake_model,
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
    )

    assert (
        result.metadata.collection_name
        == "steel_products_paraphrase-multilingual-minilm-l12-v2_20260804T123456Z"
    )
    assert result.metadata_path == metadata_path
    assert result.metadata.schema_version == "v2"
    assert result.metadata.document_count == 2
    assert result.metadata.source_row_count == 2
    assert result.metadata.deduplicated_row_count == 2
    assert result.metadata.index_schema_version == 2
    assert result.metadata.index_format_version == 1
    assert result.metadata.dataset_sha256 == result.metadata.csv_sha256
    assert result.metadata.built_at == "20260804T123456Z"
    assert result.metadata.point_count == 2
    assert result.metadata.git_commit != "unknown"
    assert metadata_path.exists()
    metadata_text = metadata_path.read_text(encoding="utf-8")
    assert "csv_sha256" in metadata_text
    assert '"dataset_sha256":' in metadata_text
    assert '"index_format_version": 1' in metadata_text
    assert '"schema_version": "v2"' in metadata_text
    assert '"index_schema_version": 2' in metadata_text

    assert len(fake_model.calls) == 6
    assert fake_model.calls[0]["texts"] == [document.semantic_text for document in documents]
    assert fake_model.calls[0]["show_progress_bar"] is True
    assert [call["texts"] for call in fake_model.calls[1:]] == [
        ["1184399"],
        ["а0486"],
        ["Temper DN80 PN16"],
        ["Broen Ду80 Ру16"],
        ["фланцевый кран Ду50 Ру40"],
    ]

    assert len(fake_client.created_collections) == 1
    assert fake_client.created_collections[0]["collection_name"] == result.metadata.collection_name
    assert len(fake_client.upserts) == 1
    assert len(fake_client.upserts[0]["points"]) == 2
    assert fake_client.upserts[0]["points"][0].id == _point_id_for(documents[0].steel_id)
    UUID(str(fake_client.upserts[0]["points"][0].id))
    assert (
        fake_client.upserts[0]["points"][0].payload["semantic_text"] == "SOURCE_SENTINEL semantic"
    )
    assert len(fake_client.upserts[0]["points"][0].vector["dense"]) == 384
    assert fake_client.upserts[0]["points"][0].vector["dense"][0] == 1.0
    assert fake_client.upserts[0]["points"][0].vector["sparse"].text == "SOURCE_SENTINEL lexical"
    assert len(fake_client.query_calls) == 5
    assert len(fake_client.update_collection_calls) == 1
    assert (
        fake_client.update_collection_calls[0]["collection_name"] == result.metadata.collection_name
    )
    assert fake_client.update_collection_calls[0]["metadata"]["point_count"] == 2
    assert len(fake_client.alias_operations) == 1
    assert len(fake_client.alias_operations[0]) == 2


def test_build_index_records_multi_source_provenance(tmp_path: Path, monkeypatch) -> None:
    csv_paths = [
        tmp_path / "mapping_results.csv",
        tmp_path / "butterfly_mapping_results.csv",
        tmp_path / "competitor_ld_mapping.csv",
    ]
    metadata_path = tmp_path / "index_build.json"
    fake_client = FakeQdrantClient()
    fake_model = FakeEmbedder(calls=[])
    documents = _make_documents()
    source_rows = _make_frame()
    source_files = [
        SourceFileRecord(
            name="mapping_results.csv",
            adapter="canonical",
            sha256="a" * 64,
            rows=2,
        ),
        SourceFileRecord(
            name="butterfly_mapping_results.csv",
            adapter="butterfly",
            sha256="b" * 64,
            rows=11648,
        ),
        SourceFileRecord(
            name="competitor_ld_mapping.csv",
            adapter="competitor_ld",
            sha256="c" * 64,
            rows=3679,
        ),
    ]

    monkeypatch.setattr(
        "rag_steel.indexer.load_source_bundle",
        lambda paths: (source_rows, source_files),
    )
    monkeypatch.setattr(
        "rag_steel.indexer.build_source_documents_from_frame",
        lambda df: documents,
    )

    result = build_index(
        csv_paths,
        embedder=fake_model,
        recreate=False,
        client=fake_client,
        metadata_path=metadata_path,
        batch_size=2,
        build_time=datetime(2026, 8, 4, 12, 34, 56, tzinfo=timezone.utc),
        smoke_queries=["Temper DN80 PN16"],
    )

    assert result.metadata.source_row_count == len(source_rows)
    assert result.metadata.deduplicated_row_count == len(source_rows.drop_duplicates())
    assert result.metadata.source_files == [record.to_dict() for record in source_files]
    assert result.metadata.dataset_sha256 == result.metadata.csv_sha256
    assert fake_client.created_collections[0]["metadata"]["source_files"] == [
        record.to_dict() for record in source_files
    ]


def test_build_index_skips_alias_switch_without_recreate(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "mapping_results.csv"
    _make_frame().to_csv(csv_path, index=False)
    fake_client = FakeQdrantClient()
    fake_model = FakeEmbedder(calls=[])
    source_rows = _make_frame()
    source_files = _make_source_files()

    monkeypatch.setattr(
        "rag_steel.indexer.load_source_bundle",
        lambda paths: (source_rows, source_files),
    )
    monkeypatch.setattr(
        "rag_steel.indexer.build_source_documents_from_frame", lambda df: _make_documents()
    )

    build_index(
        csv_path,
        embedder=fake_model,
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
    )

    assert fake_client.alias_operations == []


def test_build_index_uses_raw_texts_for_local_models(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "mapping_results.csv"
    _make_frame().to_csv(csv_path, index=False)
    metadata_path = tmp_path / "index_build.json"
    fake_client = FakeQdrantClient()
    source_rows = _make_frame()
    source_files = _make_source_files()

    class LocalFakeEmbedder(FakeEmbedder):
        model_name = "intfloat/multilingual-e5-base"

    fake_model = LocalFakeEmbedder(calls=[], dimension=768)

    monkeypatch.setattr(
        "rag_steel.indexer.load_source_bundle",
        lambda paths: (source_rows, source_files),
    )
    monkeypatch.setattr(
        "rag_steel.indexer.build_source_documents_from_frame", lambda df: _make_documents()
    )

    result = build_index(
        csv_path,
        embedder=fake_model,
        recreate=False,
        client=fake_client,
        metadata_path=metadata_path,
        batch_size=2,
        build_time=datetime(2026, 8, 4, 12, 34, 56, tzinfo=timezone.utc),
        smoke_queries=["Temper DN80 PN16"],
    )

    assert result.metadata.embedding_dimension == 768
    assert fake_model.calls[0]["texts"] == [
        "SOURCE_SENTINEL semantic",
        "SOURCE_SENTINEL semantic 2",
    ]
    assert fake_model.calls[1]["texts"] == ["Temper DN80 PN16"]


def test_build_index_uses_supplied_embedding_dimension_in_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    csv_path = tmp_path / "mapping_results.csv"
    _make_frame().to_csv(csv_path, index=False)
    fake_client = FakeQdrantClient()
    source_rows = _make_frame()
    source_files = _make_source_files()

    class WrongDimensionEmbedder(FakeEmbedder):
        pass

    monkeypatch.setattr(
        "rag_steel.indexer.load_source_bundle",
        lambda paths: (source_rows, source_files),
    )
    monkeypatch.setattr(
        "rag_steel.indexer.build_source_documents_from_frame", lambda df: _make_documents()
    )

    result = build_index(
        csv_path,
        embedder=WrongDimensionEmbedder(calls=[], dimension=999),
        recreate=False,
        client=fake_client,
        metadata_path=tmp_path / "index_build.json",
        batch_size=2,
        build_time=datetime(2026, 8, 4, 12, 34, 56, tzinfo=timezone.utc),
        smoke_queries=["Temper DN80 PN16"],
    )

    assert result.metadata.embedding_dimension == 999


def test_wait_for_qdrant_ready_retries_transient_503(monkeypatch) -> None:
    fake_client = FlakyReadyQdrantClient(failures_before_ready=2)
    sleeps: list[float] = []

    monkeypatch.setattr("rag_steel.indexer.sleep", lambda seconds: sleeps.append(seconds))

    _wait_for_qdrant_ready(
        fake_client, "steel_products_example", timeout_seconds=5.0, poll_interval_seconds=0.25
    )

    assert fake_client.collection_exists_calls == 3
    assert sleeps == [0.25, 0.25]


def test_wait_for_qdrant_ready_times_out_on_persistent_503(monkeypatch) -> None:
    fake_client = FlakyReadyQdrantClient(failures_before_ready=999)
    timeline = iter([0.0, 0.1, 0.2, 1.2, 1.3])

    monkeypatch.setattr("rag_steel.indexer.sleep", lambda _: None)
    monkeypatch.setattr("rag_steel.indexer.monotonic", lambda: next(timeline))

    with pytest.raises(RuntimeError, match="Qdrant was not ready within 1.0s"):
        _wait_for_qdrant_ready(
            fake_client, "steel_products_example", timeout_seconds=1.0, poll_interval_seconds=0.25
        )


def test_upsert_with_retry_retries_twice_before_success(monkeypatch) -> None:
    fake_client = FlakyUpsertQdrantClient(failures_before_success=2)
    sleeps: list[float] = []

    monkeypatch.setattr("rag_steel.indexer.sleep", lambda seconds: sleeps.append(seconds))

    _upsert_with_retry(
        fake_client, collection_name="steel_products_example", points=[], retry_count=2
    )

    assert fake_client.upsert_attempts == 3
    assert sleeps == [1.0, 1.0]


def test_upsert_with_retry_raises_after_exhausting_retries(monkeypatch) -> None:
    fake_client = FlakyUpsertQdrantClient(failures_before_success=3)
    sleeps: list[float] = []

    monkeypatch.setattr("rag_steel.indexer.sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(RuntimeError, match="timed out"):
        _upsert_with_retry(
            fake_client, collection_name="steel_products_example", points=[], retry_count=2
        )

    assert fake_client.upsert_attempts == 3
    assert sleeps == [1.0, 1.0]
