"""Shared index metadata constants."""

from __future__ import annotations

from typing import Any

INDEX_SCHEMA_VERSION = 2
SCHEMA_VERSION = "v2"
SUPPORTED_INDEX_FORMAT_VERSION = 1


def check_index_compatibility(
    *,
    metadata: dict[str, Any] | None,
    actual_dimension: int | None,
    settings: Any,
) -> dict[str, Any]:
    metadata = dict(metadata or {})
    expected_dimension = int(getattr(settings, "embedding_dimension", 0) or 0)
    expected_model = getattr(settings, "embedding_model", None)
    warnings: list[str] = []
    result: dict[str, Any] = {
        "compatible": True,
        "reason": None,
        "warnings": warnings,
        "schema_version": metadata.get("schema_version") or metadata.get("index_schema_version"),
        "index_format_version": metadata.get("index_format_version"),
        "embedding_model": metadata.get("embedding_model"),
        "embedding_dimension": metadata.get("embedding_dimension"),
    }

    if metadata and metadata.get("schema_version") is None:
        result["schema_version"] = (
            f"v{metadata.get('index_schema_version')}"
            if metadata.get("index_schema_version") is not None
            else None
        )

    if not metadata:
        warnings.append("INDEX_METADATA_MISSING")
    else:
        schema_version = metadata.get("schema_version")
        index_schema_version = metadata.get("index_schema_version")
        if schema_version is not None and schema_version != SCHEMA_VERSION:
            result["compatible"] = False
            result["reason"] = "SCHEMA_VERSION_MISMATCH"
            return result
        if index_schema_version is not None and int(index_schema_version) != INDEX_SCHEMA_VERSION:
            result["compatible"] = False
            result["reason"] = "INDEX_SCHEMA_VERSION_MISMATCH"
            return result

    if actual_dimension is None:
        result["compatible"] = False
        result["reason"] = "VECTOR_DIMENSION_MISSING"
        return result

    if actual_dimension != expected_dimension:
        result["compatible"] = False
        result["reason"] = "VECTOR_DIMENSION_MISMATCH"
        return result

    if metadata:
        if metadata.get("embedding_dimension") not in {None, expected_dimension}:
            result["compatible"] = False
            result["reason"] = "VECTOR_DIMENSION_MISMATCH"
            return result

        if metadata.get("embedding_model") not in {None, expected_model}:
            result["compatible"] = False
            result["reason"] = "EMBEDDING_MODEL_MISMATCH"
            return result

        index_format_version = metadata.get("index_format_version")
        if index_format_version is not None and int(index_format_version) != (
            SUPPORTED_INDEX_FORMAT_VERSION
        ):
            result["compatible"] = False
            result["reason"] = "INDEX_FORMAT_UNSUPPORTED"
            return result

    result["compatible"] = True
    return result


__all__ = [
    "INDEX_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SUPPORTED_INDEX_FORMAT_VERSION",
    "check_index_compatibility",
]
