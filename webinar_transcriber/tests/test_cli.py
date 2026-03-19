"""Tests for the CLI entrypoints."""

import runpy
import sys

from click.testing import CliRunner

from webinar_transcriber.cli import main
from webinar_transcriber.models import (
    Diagnostics,
    MediaAsset,
    MediaType,
    ReportDocument,
    TranscriptionResult,
)
from webinar_transcriber.paths import OutputDirectoryExistsError, RunLayout
from webinar_transcriber.processor import ProcessArtifacts


def test_main_help_shows_process_command() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "process" in result.output


def test_main_version_prints_package_version() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_process_command_runs_pipeline(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    input_path = tmp_path / "demo.mp4"
    input_path.write_text("stub", encoding="utf-8")
    run_dir = tmp_path / "run-dir"

    def fake_process_input(**kwargs) -> ProcessArtifacts:
        assert kwargs["input_path"] == input_path
        assert kwargs["ocr_enabled"] is True
        assert kwargs["output_format"] == "json"
        return ProcessArtifacts(
            layout=RunLayout(run_dir=run_dir),
            media_asset=MediaAsset(
                path=str(input_path),
                media_type=MediaType.VIDEO,
                duration_sec=1.0,
            ),
            transcription=TranscriptionResult(detected_language="en"),
            report=ReportDocument(
                title="Demo",
                source_file=str(input_path),
                media_type=MediaType.VIDEO,
                ocr_enabled=True,
            ),
            diagnostics=Diagnostics(),
        )

    monkeypatch.setattr("webinar_transcriber.cli.process_input", fake_process_input)

    result = runner.invoke(main, ["process", str(input_path), "--ocr", "--format", "json"])

    assert result.exit_code == 0
    assert "ocr=True" in result.output
    assert "format=json" in result.output
    assert str(run_dir) in result.output


def test_process_command_rejects_missing_input(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["process", str(tmp_path / "missing.wav")])

    assert result.exit_code != 0
    assert "Input file does not exist" in result.output


def test_process_command_rejects_existing_output_directory(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    input_path = tmp_path / "demo.wav"
    output_dir = tmp_path / "run"
    input_path.write_text("stub", encoding="utf-8")
    output_dir.mkdir()

    def should_not_run(**_: object) -> ProcessArtifacts:
        raise OutputDirectoryExistsError(f"Output directory already exists: {output_dir}")

    monkeypatch.setattr("webinar_transcriber.cli.process_input", should_not_run)

    result = runner.invoke(main, ["process", str(input_path), "--output-dir", str(output_dir)])

    assert result.exit_code != 0
    assert "Output directory already exists" in result.output


def test_module_entrypoint_reports_version() -> None:
    original_argv = sys.argv[:]
    sys.argv = ["python", "--version"]

    try:
        with CliRunner().isolated_filesystem():
            try:
                runpy.run_module("webinar_transcriber", run_name="__main__")
            except SystemExit as error:
                assert error.code == 0
    finally:
        sys.argv = original_argv
