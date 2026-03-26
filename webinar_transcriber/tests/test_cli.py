"""Tests for the CLI entrypoints."""

import json
import runpy
import sys
from unittest.mock import ANY, patch

from click.testing import CliRunner

from webinar_transcriber.asr import DEFAULT_ASR_THREADS
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
    assert "Process an audio or video input." in result.output
    assert "Extract representative frames from a video." in result.output
    assert "representative frames from a..." not in result.output


def test_main_version_prints_package_version() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "0.1.1" in result.output


def test_process_command_runs_pipeline(tmp_path) -> None:
    runner = CliRunner()
    input_path = tmp_path / "demo.mp4"
    input_path.write_text("stub", encoding="utf-8")
    run_dir = tmp_path / "run-dir"

    with patch(
        "webinar_transcriber.cli.process_input",
        return_value=ProcessArtifacts(
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
        ),
    ) as process_input_mock:
        result = runner.invoke(main, ["process", str(input_path), "--format", "json"])

    assert result.exit_code == 0
    assert result.output == ""
    process_input_mock.assert_called_once_with(
        input_path=input_path,
        output_dir=None,
        output_format="json",
        asr_model=None,
        vad_enabled=True,
        chunk_target_sec=20.0,
        chunk_max_sec=30.0,
        chunk_overlap_sec=1.5,
        asr_threads=DEFAULT_ASR_THREADS,
        enable_llm=False,
        reporter=ANY,
    )
    assert process_input_mock.call_args.kwargs["reporter"].__class__.__name__ == "RichStageReporter"


def test_process_command_forwards_asr_options(tmp_path) -> None:
    runner = CliRunner()
    input_path = tmp_path / "demo.mp4"
    input_path.write_text("stub", encoding="utf-8")
    run_dir = tmp_path / "run-dir"

    with patch(
        "webinar_transcriber.cli.process_input",
        return_value=ProcessArtifacts(
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
        ),
    ) as process_input_mock:
        result = runner.invoke(
            main,
            [
                "process",
                str(input_path),
                "--asr-model",
                "models/whisper-cpp/custom.bin",
                "--no-vad",
                "--chunk-target-sec",
                "18",
                "--chunk-max-sec",
                "24",
                "--chunk-overlap-sec",
                "2",
                "--threads",
                "6",
                "--llm",
            ],
        )

    assert result.exit_code == 0
    process_input_mock.assert_called_once_with(
        input_path=input_path,
        output_dir=None,
        output_format="all",
        asr_model="models/whisper-cpp/custom.bin",
        vad_enabled=False,
        chunk_target_sec=18.0,
        chunk_max_sec=24.0,
        chunk_overlap_sec=2.0,
        asr_threads=6,
        enable_llm=True,
        reporter=ANY,
    )


def test_process_help_describes_asr_options() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["process", "--help"])

    assert result.exit_code == 0
    assert "--asr-model" in result.output
    assert "--vad / --no-vad" in result.output
    assert "--chunk-target-sec" in result.output
    assert "--chunk-max-sec" in result.output
    assert "--chunk-overlap-sec" in result.output
    assert "--threads" in result.output
    assert "--llm" in result.output
    assert "Override the whisper.cpp model path" in result.output
    assert "--asr-compute-type" not in result.output
    assert "--llm-model" not in result.output


def test_process_command_rejects_missing_input(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["process", str(tmp_path / "missing.wav")])

    assert result.exit_code != 0
    assert "Input file does not exist" in result.output


def test_process_command_colors_top_level_errors(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["process", str(tmp_path / "missing.wav")], color=True)

    assert result.exit_code != 0
    assert "\x1b[" in result.output
    assert "Error:" in result.output


def test_process_command_rejects_existing_output_directory(tmp_path) -> None:
    runner = CliRunner()
    input_path = tmp_path / "demo.wav"
    output_dir = tmp_path / "run"
    input_path.write_text("stub", encoding="utf-8")
    output_dir.mkdir()

    with patch(
        "webinar_transcriber.cli.process_input",
        side_effect=OutputDirectoryExistsError(f"Output directory already exists: {output_dir}"),
    ):
        result = runner.invoke(main, ["process", str(input_path), "--output-dir", str(output_dir)])

    assert result.exit_code != 0
    assert "Output directory already exists" in result.output


def test_process_command_handles_ctrl_c(tmp_path) -> None:
    runner = CliRunner()
    input_path = tmp_path / "demo.wav"
    input_path.write_text("stub", encoding="utf-8")

    with patch(
        "webinar_transcriber.cli.process_input",
        side_effect=KeyboardInterrupt,
    ):
        result = runner.invoke(main, ["process", str(input_path)])

    assert result.exit_code == 130
    assert "Interrupted" in result.output


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


def test_extract_frames_command_handles_ctrl_c(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    input_path = tmp_path / "demo.mp4"
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
            media_type=MediaType.VIDEO,
            duration_sec=2.0,
        ),
    )
    monkeypatch.setattr("webinar_transcriber.cli.estimate_sample_count", lambda _duration: 2)

    def interrupted_detect_scenes(*_args, **_kwargs) -> list[Scene]:
        raise KeyboardInterrupt

    monkeypatch.setattr("webinar_transcriber.cli.detect_scenes", interrupted_detect_scenes)

    result = runner.invoke(main, ["extract-frames", str(input_path)])

    assert result.exit_code == 130
    assert "Interrupted" in result.output


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
