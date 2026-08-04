import importlib


def test_main_import_is_ready() -> None:
    importlib.import_module("main")
