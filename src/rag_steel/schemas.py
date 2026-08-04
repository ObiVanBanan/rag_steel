"""Pydantic schemas for grouped source and LD products."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LDProduct(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article: str
    article_norm: str
    name: str
    url: str | None = None
    dn: float | None = None
    pn_bar: float | None = None
    connection: str | None = None
    medium: str | None = None
    control: str | None = None
    temperature: str | None = None
    length_mm: float | None = None
    price: float | None = None


class SteelProductDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steel_id: str
    article: str
    article_norm: str
    article_compact: str
    name: str
    name_variants: list[str] = Field(default_factory=list)
    brand: str | None = None
    dn: float | None = None
    pn_bar: float | None = None
    connection: str | None = None
    medium: str | None = None
    control: str | None = None
    temperature: str | None = None
    length_mm: float | None = None
    url: str | None = None
    semantic_text: str = ""
    lexical_text: str = ""
    ld_candidates: list[LDProduct] = Field(default_factory=list)

