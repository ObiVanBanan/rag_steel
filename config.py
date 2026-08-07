"""Compatibility wrapper for package configuration."""

from __future__ import annotations

from importlib import import_module, reload

_impl = reload(import_module("rag_steel.config"))

for _name in getattr(_impl, "__all__", []):
    globals()[_name] = getattr(_impl, _name)

__all__ = list(getattr(_impl, "__all__", []))
