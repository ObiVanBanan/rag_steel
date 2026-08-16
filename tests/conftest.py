from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP_ROOT = Path.cwd() / ".tmp" / "pytest"
_TMP_ROOT.mkdir(parents=True, exist_ok=True)

for name in ("TMPDIR", "TEMP", "TMP"):
    os.environ[name] = str(_TMP_ROOT)

tempfile.tempdir = str(_TMP_ROOT)
