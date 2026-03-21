"""Tests for the CLI entrypoints."""

import json
import runpy
import sys

from click.testing import CliRunner

from webinar_transcriber.cli import main
from webinar_transcriber.models import (
    Diagnostics,
    MediaAsset,
    MediaType,
    ReportDocument,
    Scene,
    SlideFrame,
    TranscriptionResult,
)
from webinar_transcriber.paths import OutputDirectoryExistsError, RunLayout
from webinar_transcriber.processor import ProcessArtifacts


def test_main_help_shows_process_command() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "process" in result.output
    assert "extract-frames" in result.output


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
        assert kwargs["output_format"] == "json"
        assert kwargs["asr_backend"] == "auto"
        assert kwargs["asr_model"] is None
        assert kwargs["reporter"].__class__.__name__ == "RichStageReporter"
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
            ),
            diagnostics=Diagnostics(),
        )

    monkeypatch.setattr("webinar_transcriber.cli.process_input", fake_process_input)

    result = runner.invoke(main, ["process", str(input_path), "--format", "json"])

    assert result.exit_code == 0
    assert "format=json" in result.output
    assert str(run_dir) in result.output


def test_process_command_forwards_asr_options(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    input_path = tmp_path / "demo.mp4"
    input_path.write_text("stub", encoding="utf-8")
    run_dir = tmp_path / "run-dir"

    def fake_process_input(**kwargs) -> ProcessArtifacts:
        assert kwargs["asr_backend"] == "faster-whisper"
        assert kwargs["asr_model"] == "small"
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
            ),
            diagnostics=Diagnostics(),
        )

    monkeypatch.setattr("webinar_transcriber.cli.process_input", fake_process_input)

    result = runner.invoke(
        main,
        [
            "process",
            str(input_path),
            "--asr-backend",
            "faster-whisper",
            "--asr-model",
            "small",
        ],
    )

    assert result.exit_code == 0


def test_process_help_describes_asr_options() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["process", "--help"])

    assert result.exit_code == 0
    assert "--asr-backend" in result.output
    assert "--asr-model" in result.output
    assert "MLX repo name" in result.output
    assert "--asr-compute-type" not in result.output


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


def test_extract_frames_command_writes_scene_artifacts(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    input_path = tmp_path / "demo.mp4"
    input_path.write_text("stub", encoding="utf-8")
    run_dir = tmp_path / "frames-run"
    frame_path = run_dir / "frames" / "scene-1.png"
    run_dir.mkdir()
    (run_dir / "frames").mkdir()
    frame_path.write_text("png", encoding="utf-8")

    scenes = [Scene(id="scene-1", start_sec=0.0, end_sec=2.0)]

    monkeypatch.setattr(
        "webinar_transcriber.cli.create_run_layout",
        lambda **kwargs: RunLayout(run_dir=run_dir),
    )
    monkeypatch.setattr(
        "webinar_transcriber.cli.probe_media",
        lambda _path: MediaAsset(
            path=str(input_path),
            media_type=MediaType.VIDEO,
            duration_sec=2.0,
        ),
    )
    monkeypatch.setattr("webinar_transcriber.cli.estimate_sample_count", lambda _duration: 2)
    monkeypatch.setattr(
        "webinar_transcriber.cli.detect_scenes",
        lambda *_args, **_kwargs: scenes,
    )
    monkeypatch.setattr(
        "webinar_transcriber.cli.extract_representative_frames",
        lambda *_args, **_kwargs: [
            SlideFrame(
                id="frame-1",
                scene_id="scene-1",
                image_path=str(frame_path),
                timestamp_sec=1.0,
            )
        ],
    )

    result = runner.invoke(main, ["extract-frames", str(input_path)])

    assert result.exit_code == 0
    assert "Extracted 1 frames into" in result.output
    scenes_payload = json.loads((run_dir / "scenes.json").read_text(encoding="utf-8"))
    assert scenes_payload["scenes"][0]["id"] == "scene-1"


def test_extract_frames_command_rejects_audio_input(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    input_path = tmp_path / "demo.wav"
    input_path.write_text("stub", encoding="utf-8")
    run_dir = tmp_path / "frames-run"
    run_dir.mkdir()

    monkeypatch.setattr(
        "webinar_transcriber.cli.create_run_layout",
        lambda **kwargs: RunLayout(run_dir=run_dir),
    )
    monkeypatch.setattr(
        "webinar_transcriber.cli.probe_media",
        lambda _path: MediaAsset(
            path=str(input_path),
            media_type=MediaType.AUDIO,
            duration_sec=2.0,
        ),
    )

    result = runner.invoke(main, ["extract-frames", str(input_path)])

    assert result.exit_code != 0
    assert "only supported for video input" in result.output


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
