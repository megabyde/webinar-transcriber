"""Tests for package version resolution."""

import importlib.metadata

import webinar_transcriber


class TestResolveVersion:
    def test_uses_generated_version_file_first(self, monkeypatch) -> None:
        monkeypatch.setattr(webinar_transcriber, "_source_version", "2.0.0")
        monkeypatch.setattr(webinar_transcriber, "version", lambda _name: "1.2.3")

        assert webinar_transcriber.get_version() == "2.0.0"

    def test_falls_back_to_installed_distribution_version(self, monkeypatch) -> None:
        monkeypatch.setattr(webinar_transcriber, "_source_version", None)
        monkeypatch.setattr(webinar_transcriber, "version", lambda _name: "1.2.3")

        assert webinar_transcriber.get_version() == "1.2.3"

    def test_returns_zero_when_no_source_or_distribution_version(self, monkeypatch) -> None:
        def raise_missing_distribution(_name: str) -> str:
            raise importlib.metadata.PackageNotFoundError("webinar-transcriber")

        monkeypatch.setattr(webinar_transcriber, "_source_version", None)
        monkeypatch.setattr(webinar_transcriber, "version", raise_missing_distribution)

        assert webinar_transcriber.get_version() == "0.0.0"
