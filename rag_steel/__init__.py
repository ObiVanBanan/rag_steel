"""Bridge package for local, non-installed runs.

This project keeps the implementation under ``src/rag_steel`` while
``[tool.uv].package = false`` means commands like ``uvicorn main:app`` do not
install the package or add ``src`` to ``sys.path``. Extending ``__path__``
lets ``import rag_steel...`` work from the repository root as well.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_SRC_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "src" / "rag_steel"
if _SRC_PACKAGE_DIR.is_dir():
    __path__.append(str(_SRC_PACKAGE_DIR))

