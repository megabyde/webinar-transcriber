"""Tests for package version resolution."""

import importlib
from types import SimpleNamespace

import webinar_transcriber


class TestResolveVersion:
    def test_uses_generated_version_file(self, monkeypatch) -> None:
        monkeypatch.setattr(
            importlib, "import_module", lambda _name: SimpleNamespace(__version__="1.2.3")
        )

        assert importlib.reload(webinar_transcriber).__version__ == "1.2.3"

    def test_falls_back_without_generated_version_file(self, monkeypatch) -> None:
        def raise_missing_module(_name: str) -> SimpleNamespace:
            raise ModuleNotFoundError("webinar_transcriber._version")

        monkeypatch.setattr(importlib, "import_module", raise_missing_module)

        assert importlib.reload(webinar_transcriber).__version__ == "0.0.0"
