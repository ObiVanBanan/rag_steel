"""Pydantic schemas for the V4 evaluation suite."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ExpectedAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_brand: str | None = None
    resolved_brand: str | None = None

    article: str | None = None
    resolved_article: str | None = None

    dn: float | None = None
    pn_bar: float | None = None
    connection: str | None = None

    body_material: str | None = None
    medium: str | None = None
    control: str | None = None
    temperature: str | None = None
    length_mm: float | None = None
    series: str | None = None


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    query: str
    expected_status: str
    expected_resolution_mode: str
    expected_attributes: ExpectedAttributes
    eligible_competitor_articles: list[str] = Field(default_factory=list)
    preferred_competitor_articles: list[str] = Field(default_factory=list)
    expected_ld_articles_by_competitor: dict[str, list[str]] = Field(default_factory=dict)
