from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp"
_TMP_ROOT.mkdir(parents=True, exist_ok=True)

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for candidate in (_ROOT, _SRC):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

for name in ("TMPDIR", "TEMP", "TMP"):
    os.environ[name] = str(_TMP_ROOT)

tempfile.tempdir = str(_TMP_ROOT)
