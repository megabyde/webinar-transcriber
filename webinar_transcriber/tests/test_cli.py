"""Tests for the CLI entrypoints."""

import runpy
import sys
from unittest.mock import ANY, patch

import pytest
from click.testing import CliRunner

from webinar_transcriber import __version__
from webinar_transcriber.asr import ASRProcessingError, PromptCarryoverSettings
from webinar_transcriber.cli import main
from webinar_transcriber.llm.contracts import LLMConfigurationError, LLMProcessingError
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


def _process_artifacts(input_path, run_dir) -> ProcessArtifacts:
    return ProcessArtifacts(
        layout=RunLayout(run_dir=run_dir),
        media_asset=VideoAsset(path=str(input_path), duration_sec=1.0),
        transcription=TranscriptionResult(detected_language="en"),
        report=ReportDocument(
            title="Demo", source_file=str(input_path), media_type=MediaType.VIDEO
        ),
        diagnostics=Diagnostics(),
    )


class TestCli:
    def test_main_help_describes_root_command(self) -> None:
        runner = CliRunner()

        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Transcribe an audio or video input file." in result.output

    def test_main_version_prints_package_version(self) -> None:
        runner = CliRunner()

        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert result.output == f"webinar-transcriber, version {__version__}\n"

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

    def test_runs_pipeline(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.mp4"
        input_path.write_text("stub", encoding="utf-8")
        run_dir = tmp_path / "run-dir"

        with (
            patch(
                "webinar_transcriber.cli.process_input",
                return_value=_process_artifacts(input_path, run_dir),
            ) as process_input_mock,
            patch("webinar_transcriber.cli.default_asr_threads", return_value=7),
        ):
            result = runner.invoke(main, [str(input_path)])

        assert result.exit_code == 0
        assert result.output == ""
        process_input_mock.assert_called_once_with(
            input_path=input_path,
            output_dir=None,
            asr_model=None,
            language=None,
            vad=VadSettings(),
            carryover=PromptCarryoverSettings(),
            asr_threads=7,
            keep_audio=None,
            enable_llm=False,
            reporter=ANY,
        )
        assert process_input_mock.call_args.kwargs["reporter"].__class__.__name__ == (
            "RichStageReporter"
        )

    def test_forwards_asr_options(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.mp4"
        input_path.write_text("stub", encoding="utf-8")
        run_dir = tmp_path / "run-dir"

        with (
            patch(
                "webinar_transcriber.cli.process_input",
                return_value=_process_artifacts(input_path, run_dir),
            ) as process_input_mock,
            patch("webinar_transcriber.cli.default_asr_threads", return_value=6),
        ):
            result = runner.invoke(
                main,
                [
                    str(input_path),
                    "--asr-model",
                    "models/whisper-cpp/custom.bin",
                    "--language",
                    "en",
                    "--threads",
                    "3",
                    "--no-vad",
                    "--keep-audio",
                    "mp3",
                    "--llm",
                ],
            )

        assert result.exit_code == 0
        process_input_mock.assert_called_once_with(
            input_path=input_path,
            output_dir=None,
            asr_model="models/whisper-cpp/custom.bin",
            language="en",
            vad=VadSettings(enabled=False),
            carryover=PromptCarryoverSettings(),
            asr_threads=3,
            keep_audio="mp3",
            enable_llm=True,
            reporter=ANY,
        )

    def test_keep_audio_without_format_defaults_to_mp3(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.mp4"
        input_path.write_text("stub", encoding="utf-8")
        run_dir = tmp_path / "run-dir"

        with patch(
            "webinar_transcriber.cli.process_input",
            return_value=_process_artifacts(input_path, run_dir),
        ) as process_input_mock:
            result = runner.invoke(main, [str(input_path), "--keep-audio"])

        assert result.exit_code == 0
        assert process_input_mock.call_args.kwargs["keep_audio"] == "mp3"

    def test_help_describes_processing_options(self) -> None:
        runner = CliRunner()

        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "--asr-model" in result.output
        assert "--language" in result.output
        assert "--threads" in result.output
        assert "--vad / --no-vad" in result.output
        assert "--keep-audio" in result.output
        assert "--llm" in result.output
        assert "Override the whisper.cpp model identifier" in result.output
        assert "model path" in result.output
        assert "provider-backed report" in result.output
        assert "enhancement." in result.output

    @pytest.mark.parametrize(
        ("path_name", "create_directory", "message"),
        [
            ("missing.wav", False, "Input file does not exist"),
            ("input-dir", True, "Input path is not a file"),
        ],
    )
    def test_rejects_invalid_input(
        self, tmp_path, path_name: str, create_directory: bool, message: str
    ) -> None:
        runner = CliRunner()
        input_path = tmp_path / path_name
        if create_directory:
            input_path.mkdir()

        result = runner.invoke(main, [str(input_path)])

        assert result.exit_code != 0
        assert message in result.output

    def test_rejects_invalid_thread_count(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.wav"
        input_path.write_text("stub", encoding="utf-8")

        result = runner.invoke(main, [str(input_path), "--threads", "0"])

        assert result.exit_code != 0
        assert "Invalid value for '--threads'" in result.output

    def test_rejects_existing_output_directory(self, tmp_path) -> None:
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

    @pytest.mark.parametrize(
        ("error", "message"),
        [
            (ASRProcessingError("missing ASR model"), "missing ASR model"),
            (LLMConfigurationError("missing LLM config"), "missing LLM config"),
            (LLMProcessingError("LLM request failed"), "LLM request failed"),
        ],
    )
    def test_reports_expected_runtime_errors_as_cli_errors(
        self, tmp_path, error: Exception, message: str
    ) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.wav"
        input_path.write_text("stub", encoding="utf-8")

        with patch("webinar_transcriber.cli.process_input", side_effect=error):
            result = runner.invoke(main, [str(input_path)])

        assert result.exit_code != 0
        assert message in result.output

    def test_resets_active_display_before_cli_errors(self, tmp_path) -> None:
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

    def test_handles_ctrl_c(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.wav"
        input_path.write_text("stub", encoding="utf-8")

        class FakeReporter:
            def interrupted(self) -> None:
                print("Interrupted")

        with (
            patch("webinar_transcriber.cli.process_input", side_effect=KeyboardInterrupt),
            patch("webinar_transcriber.cli.RichStageReporter", return_value=FakeReporter()),
        ):
            result = runner.invoke(main, [str(input_path)])

        assert result.exit_code == 130
        assert "Interrupted" in result.output
