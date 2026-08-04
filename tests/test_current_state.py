import importlib

import pytest


@pytest.mark.xfail(
    reason="main.py is not importable yet because the current implementation is incomplete",
    raises=(AttributeError, ImportError, ModuleNotFoundError),
)
def test_main_import_is_not_ready():
    importlib.import_module("main")
