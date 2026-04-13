"""Tests for the CLI entrypoints."""

import runpy
import sys
from unittest.mock import ANY, patch

import pytest
from click.testing import CliRunner

from webinar_transcriber import __version__
from webinar_transcriber.asr import DEFAULT_ASR_THREADS, PromptCarryoverSettings
from webinar_transcriber.cli import main
from webinar_transcriber.models import (
    Diagnostics,
    MediaType,
    ReportDocument,
    TranscriptionResult,
    VideoAsset,
)
from webinar_transcriber.paths import OutputDirectoryExistsError, RunLayout
from webinar_transcriber.processor import ProcessArtifacts
from webinar_transcriber.segmentation import VadSettings


class TestMainCli:
    def test_main_help_describes_root_command(self) -> None:
        runner = CliRunner()

        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Process an audio or video input file." in result.output
        assert "Commands:" not in result.output

    def test_main_version_prints_package_version(self) -> None:
        runner = CliRunner()

        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert __version__ in result.output

    def test_module_entrypoint_reports_version(self) -> None:
        original_argv = sys.argv[:]
        sys.argv = ["python", "--version"]

        try:
            with CliRunner().isolated_filesystem():
                with pytest.raises(SystemExit) as error:
                    runpy.run_module("webinar_transcriber", run_name="__main__")
                assert error.value.code == 0
        finally:
            sys.argv = original_argv


class TestProcessCommand:
    def test_process_command_runs_pipeline(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.mp4"
        input_path.write_text("stub", encoding="utf-8")
        run_dir = tmp_path / "run-dir"

        with patch(
            "webinar_transcriber.cli.process_input",
            return_value=ProcessArtifacts(
                layout=RunLayout(run_dir=run_dir),
                media_asset=VideoAsset(path=str(input_path), duration_sec=1.0),
                transcription=TranscriptionResult(detected_language="en"),
                report=ReportDocument(
                    title="Demo", source_file=str(input_path), media_type=MediaType.VIDEO
                ),
                diagnostics=Diagnostics(),
            ),
        ) as process_input_mock:
            result = runner.invoke(main, [str(input_path)])

        assert result.exit_code == 0
        assert result.output == ""
        process_input_mock.assert_called_once_with(
            input_path=input_path,
            output_dir=None,
            asr_model=None,
            vad=VadSettings(),
            carryover=PromptCarryoverSettings(),
            asr_threads=DEFAULT_ASR_THREADS,
            keep_audio=False,
            kept_audio_format="wav",
            enable_llm=False,
            reporter=ANY,
        )
        assert process_input_mock.call_args.kwargs["reporter"].__class__.__name__ == (
            "RichStageReporter"
        )

    def test_process_command_forwards_asr_options(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.mp4"
        input_path.write_text("stub", encoding="utf-8")
        run_dir = tmp_path / "run-dir"

        with patch(
            "webinar_transcriber.cli.process_input",
            return_value=ProcessArtifacts(
                layout=RunLayout(run_dir=run_dir),
                media_asset=VideoAsset(path=str(input_path), duration_sec=1.0),
                transcription=TranscriptionResult(detected_language="en"),
                report=ReportDocument(
                    title="Demo", source_file=str(input_path), media_type=MediaType.VIDEO
                ),
                diagnostics=Diagnostics(),
            ),
        ) as process_input_mock:
            result = runner.invoke(
                main,
                [
                    str(input_path),
                    "--asr-model",
                    "models/whisper-cpp/custom.bin",
                    "--no-vad",
                    "--vad-threshold",
                    "0.42",
                    "--min-speech-ms",
                    "300",
                    "--min-silence-ms",
                    "500",
                    "--speech-region-pad-ms",
                    "220",
                    "--no-carryover",
                    "--carryover-max-sentences",
                    "1",
                    "--carryover-max-tokens",
                    "32",
                    "--threads",
                    "6",
                    "--keep-audio",
                    "--audio-format",
                    "mp3",
                    "--llm",
                ],
            )

        assert result.exit_code == 0
        process_input_mock.assert_called_once_with(
            input_path=input_path,
            output_dir=None,
            asr_model="models/whisper-cpp/custom.bin",
            vad=VadSettings(
                enabled=False,
                threshold=0.42,
                min_speech_duration_ms=300,
                min_silence_duration_ms=500,
                speech_region_pad_ms=220,
            ),
            carryover=PromptCarryoverSettings(enabled=False, max_sentences=1, max_tokens=32),
            asr_threads=6,
            keep_audio=True,
            kept_audio_format="mp3",
            enable_llm=True,
            reporter=ANY,
        )

    def test_help_describes_processing_options(self) -> None:
        runner = CliRunner()

        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "--asr-model" in result.output
        assert "--vad / --no-vad" in result.output
        assert "--vad-threshold" in result.output
        assert "--min-speech-ms" in result.output
        assert "--min-silence-ms" in result.output
        assert "--speech-region-pad-ms" in result.output
        assert "--carryover / --no-carryover" in result.output
        assert "--carryover-max-sentences" in result.output
        assert "--carryover-max-tokens" in result.output
        assert "--threads" in result.output
        assert "--keep-audio / --no-keep-audio" in result.output
        assert "--audio-format" in result.output
        assert "--llm" in result.output
        assert "Override the whisper.cpp model path" in result.output
        assert "provider-backed report" in result.output
        assert "enhancement." in result.output

    @pytest.mark.parametrize(
        ("path_name", "create_directory", "message"),
        [
            ("missing.wav", False, "Input file does not exist"),
            ("input-dir", True, "Input path is not a file"),
        ],
    )
    def test_process_command_rejects_invalid_input(
        self, tmp_path, path_name: str, create_directory: bool, message: str
    ) -> None:
        runner = CliRunner()
        input_path = tmp_path / path_name
        if create_directory:
            input_path.mkdir()

        result = runner.invoke(main, [str(input_path)])

        assert result.exit_code != 0
        assert message in result.output

    def test_process_command_colors_top_level_errors(self, tmp_path) -> None:
        runner = CliRunner()

        result = runner.invoke(main, [str(tmp_path / "missing.wav")], color=True)

        assert result.exit_code != 0
        assert "\x1b[" in result.output
        assert "Error:" in result.output

    def test_process_command_rejects_existing_output_directory(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.wav"
        output_dir = tmp_path / "run"
        input_path.write_text("stub", encoding="utf-8")
        output_dir.mkdir()

        with patch(
            "webinar_transcriber.cli.process_input",
            side_effect=OutputDirectoryExistsError(
                f"Output directory already exists: {output_dir}"
            ),
        ):
            result = runner.invoke(main, [str(input_path), "--output-dir", str(output_dir)])

        assert result.exit_code != 0
        assert "Output directory already exists" in result.output

    def test_process_command_resets_active_display_before_cli_errors(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.wav"
        input_path.write_text("stub", encoding="utf-8")

        with (
            patch(
                "webinar_transcriber.cli.process_input",
                side_effect=OutputDirectoryExistsError("Output directory already exists: demo"),
            ),
            patch("webinar_transcriber.cli.RichStageReporter.reset_active_display") as reset_mock,
        ):
            result = runner.invoke(main, [str(input_path)])

        assert result.exit_code != 0
        assert "Output directory already exists: demo" in result.output
        reset_mock.assert_called_once()

    def test_process_command_handles_ctrl_c(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.wav"
        input_path.write_text("stub", encoding="utf-8")

        with patch("webinar_transcriber.cli.process_input", side_effect=KeyboardInterrupt):
            result = runner.invoke(main, [str(input_path)])

        assert result.exit_code == 130
        assert "Interrupted" in result.output
