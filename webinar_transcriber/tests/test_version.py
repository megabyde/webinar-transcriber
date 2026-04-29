"""Tests for package version resolution."""

import importlib.metadata

import webinar_transcriber


class TestResolveVersion:
    def test_uses_installed_distribution_version(self, monkeypatch) -> None:
        monkeypatch.setattr(webinar_transcriber, "version", lambda _name: "1.2.3")

        assert webinar_transcriber.get_version() == "1.2.3"

    def test_falls_back_to_generated_version_file(self, monkeypatch) -> None:
        def raise_missing_distribution(_name: str) -> str:
            raise importlib.metadata.PackageNotFoundError("webinar-transcriber")

        monkeypatch.setattr(webinar_transcriber, "version", raise_missing_distribution)
        monkeypatch.setattr(webinar_transcriber, "_source_version", "1.2.3")

        assert webinar_transcriber.get_version() == "1.2.3"
