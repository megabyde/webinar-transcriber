"""Tests for package version resolution."""

import builtins
import importlib
import importlib.metadata
import sys
from types import ModuleType

import webinar_transcriber


class TestResolveVersion:
    def test_uses_installed_distribution_version(self, monkeypatch) -> None:
        monkeypatch.setattr(importlib.metadata, "version", lambda _name: "1.2.3")

        reloaded = importlib.reload(webinar_transcriber)

        assert reloaded.__version__ == "1.2.3"

    def test_falls_back_to_generated_version_file(self, monkeypatch) -> None:
        def raise_missing_distribution(_name: str) -> str:
            raise importlib.metadata.PackageNotFoundError

        version_module = ModuleType("webinar_transcriber._version")
        version_module.__dict__["__version__"] = "1.2.3"
        monkeypatch.setattr(importlib.metadata, "version", raise_missing_distribution)
        monkeypatch.setitem(sys.modules, "webinar_transcriber._version", version_module)

        reloaded = importlib.reload(webinar_transcriber)

        assert reloaded.__version__ == "1.2.3"

    def test_falls_back_without_generated_version_file(self, monkeypatch) -> None:
        def raise_missing_distribution(_name: str) -> str:
            raise importlib.metadata.PackageNotFoundError

        real_import = builtins.__import__

        def fake_import(name, global_vars=None, local_vars=None, fromlist=(), level=0):
            if name == "webinar_transcriber._version":
                raise ModuleNotFoundError(name)
            return real_import(name, global_vars, local_vars, fromlist, level)

        monkeypatch.setattr(importlib.metadata, "version", raise_missing_distribution)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(sys.modules, "webinar_transcriber._version", raising=False)

        reloaded = importlib.reload(webinar_transcriber)

        assert reloaded.__version__ == "0.0.0"
