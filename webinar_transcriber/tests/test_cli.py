"""Tests for the bootstrap CLI."""

import runpy
import sys

from click.testing import CliRunner

from webinar_transcriber.cli import main


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


def test_process_command_echoes_bootstrap_state(tmp_path) -> None:
    runner = CliRunner()
    input_path = tmp_path / "demo.mp4"
    input_path.write_text("stub", encoding="utf-8")

    result = runner.invoke(main, ["process", str(input_path), "--ocr", "--format", "json"])

    assert result.exit_code == 0
    assert "ocr=True" in result.output
    assert "format=json" in result.output
    assert "Prepared run directory" in result.output


def test_process_command_rejects_missing_input(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["process", str(tmp_path / "missing.wav")])

    assert result.exit_code != 0
    assert "Input file does not exist" in result.output


def test_process_command_rejects_existing_output_directory(tmp_path) -> None:
    runner = CliRunner()
    input_path = tmp_path / "demo.wav"
    output_dir = tmp_path / "run"
    input_path.write_text("stub", encoding="utf-8")
    output_dir.mkdir()

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
