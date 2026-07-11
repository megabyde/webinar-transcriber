"""Tests for the CLI entrypoints."""

import runpy
import sys
from unittest.mock import ANY, patch

import pytest
from click.testing import CliRunner

from webinar_transcriber import __version__
from webinar_transcriber.asr import AsrProcessingError, default_asr_threads
from webinar_transcriber.cli import main
from webinar_transcriber.llm import LlmConfigurationError, LlmProcessingError
from webinar_transcriber.paths import OutputDirectoryExistsError
from webinar_transcriber.tests.conftest import process_artifacts


class TestCli:
    def test_main_help_describes_root_command(self) -> None:
        runner = CliRunner()

        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Transcribe one or more audio or video input files." in result.output

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
                with pytest.raises(SystemExit) as ex:
                    runpy.run_module("webinar_transcriber", run_name="__main__")
                assert ex.value.code == 0
        finally:
            sys.argv = original_argv

    def test_runs_pipeline(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.mp4"
        input_path.write_text("stub", encoding="utf-8")
        run_dir = tmp_path / "run-dir"

        with patch(
            "webinar_transcriber.cli.process_input",
            return_value=process_artifacts(input_path, run_dir),
        ) as process_input_mock:
            result = runner.invoke(main, [str(input_path)])

        assert result.exit_code == 0
        assert result.output == ""
        process_input_mock.assert_called_once_with(
            input_path=input_path,
            output_dir=None,
            threads=default_asr_threads(),
            keep_audio=False,
            llm_processor=None,
            diarizer=None,
            diarization_speaker_count=None,
            transcriber=ANY,
            reporter=ANY,
        )
        assert process_input_mock.call_args.kwargs["reporter"].__class__.__name__ == (
            "StageReporter"
        )
        assert process_input_mock.call_args.kwargs["transcriber"].model_name == "large-v3-turbo"

    def test_runs_multiple_inputs_sequentially(self, tmp_path) -> None:
        runner = CliRunner()
        first_input = tmp_path / "first.mp4"
        second_input = tmp_path / "second.mp4"
        first_input.write_text("stub", encoding="utf-8")
        second_input.write_text("stub", encoding="utf-8")
        run_dir = tmp_path / "run-dir"

        with patch(
            "webinar_transcriber.cli.process_input",
            return_value=process_artifacts(first_input, run_dir),
        ) as process_input_mock:
            result = runner.invoke(main, [str(first_input), str(second_input)])

        assert result.exit_code == 0
        assert [call.kwargs["input_path"] for call in process_input_mock.call_args_list] == [
            first_input,
            second_input,
        ]
        assert all(call.kwargs["output_dir"] is None for call in process_input_mock.call_args_list)
        assert process_input_mock.call_count == 2

    def test_forwards_asr_options(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.mp4"
        input_path.write_text("stub", encoding="utf-8")
        run_dir = tmp_path / "run-dir"
        llm_processor = object()
        diarizer = object()

        with (
            patch(
                "webinar_transcriber.cli.process_input",
                return_value=process_artifacts(input_path, run_dir),
            ) as process_input_mock,
            patch(
                "webinar_transcriber.cli.build_llm_processor_from_env", return_value=llm_processor
            ) as build_llm_processor_mock,
            patch(
                "webinar_transcriber.cli.SherpaOnnxDiarizer", return_value=diarizer
            ) as diarizer_mock,
            patch(
                "webinar_transcriber.cli.WhisperCppTranscriber", return_value=object()
            ) as transcriber_mock,
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
                    "--keep-audio",
                    "--llm",
                    "--diarize",
                    "--diarize-speakers",
                    "4",
                ],
            )

        assert result.exit_code == 0
        process_input_mock.assert_called_once_with(
            input_path=input_path,
            output_dir=None,
            threads=3,
            keep_audio=True,
            llm_processor=llm_processor,
            diarizer=diarizer,
            diarization_speaker_count=4,
            transcriber=ANY,
            reporter=ANY,
        )
        transcriber_mock.assert_called_once_with(
            model_name="models/whisper-cpp/custom.bin", threads=3, language="en"
        )
        build_llm_processor_mock.assert_called_once_with(threads=3)
        diarizer_mock.assert_called_once_with(threads=3)

    def test_keep_audio_keeps_mp3(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.mp4"
        input_path.write_text("stub", encoding="utf-8")
        run_dir = tmp_path / "run-dir"

        with patch(
            "webinar_transcriber.cli.process_input",
            return_value=process_artifacts(input_path, run_dir),
        ) as process_input_mock:
            result = runner.invoke(main, [str(input_path), "--keep-audio"])

        assert result.exit_code == 0
        assert process_input_mock.call_args.kwargs["keep_audio"]

    def test_help_describes_processing_options(self) -> None:
        runner = CliRunner()

        result = runner.invoke(main, ["--help"])
        normalized_output = " ".join(result.output.split())

        assert result.exit_code == 0
        assert "--asr-model" in result.output
        assert "--language" in result.output
        assert "--threads" in result.output
        assert "Defaults to the host CPU count, capped at 8" in normalized_output
        assert f"[default: {default_asr_threads()}; x>=1]" in normalized_output
        assert "--keep-audio" in result.output
        assert "Keep normalized transcription audio as mp3" in result.output
        assert "--llm" in result.output
        assert "--diarize / --no-diarize" in result.output
        assert "--diarize-speakers" in result.output
        assert "Override the whisper.cpp model identifier" in result.output
        assert "[default: large-v3-turbo]" in result.output
        assert "Force a Whisper language code hint" in result.output
        assert "model path" in result.output
        assert "provider-backed report" in result.output
        assert "enhancement." in result.output

    def test_rejects_invalid_thread_count(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.wav"
        input_path.write_text("stub", encoding="utf-8")

        result = runner.invoke(main, [str(input_path), "--threads", "0"])

        assert result.exit_code != 0
        assert "Invalid value for '--threads'" in result.output

    def test_rejects_output_dir_with_multiple_inputs(self, tmp_path) -> None:
        runner = CliRunner()
        first_input = tmp_path / "first.wav"
        second_input = tmp_path / "second.wav"
        first_input.write_text("stub", encoding="utf-8")
        second_input.write_text("stub", encoding="utf-8")

        with patch("webinar_transcriber.cli.process_input") as process_input_mock:
            result = runner.invoke(
                main, [str(first_input), str(second_input), "--output-dir", str(tmp_path / "run")]
            )

        assert result.exit_code != 0
        assert "--output-dir can only be used with one input file" in result.output
        process_input_mock.assert_not_called()

    def test_rejects_speaker_count_without_diarization(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.wav"
        input_path.write_text("stub", encoding="utf-8")

        with patch("webinar_transcriber.cli.process_input") as process_input_mock:
            result = runner.invoke(main, [str(input_path), "--diarize-speakers", "2"])

        assert result.exit_code != 0
        assert "--diarize-speakers requires --diarize" in result.output
        process_input_mock.assert_not_called()

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
            (AsrProcessingError("missing ASR model"), "missing ASR model"),
            (LlmConfigurationError("missing LLM config"), "missing LLM config"),
            (LlmProcessingError("LLM request failed"), "LLM request failed"),
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

    def test_reports_llm_configuration_errors_before_pipeline(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.wav"
        input_path.write_text("stub", encoding="utf-8")

        with (
            patch(
                "webinar_transcriber.cli.build_llm_processor_from_env",
                side_effect=LlmConfigurationError("missing LLM config"),
            ),
            patch("webinar_transcriber.cli.process_input") as process_input_mock,
        ):
            result = runner.invoke(main, [str(input_path), "--llm"])

        assert result.exit_code != 0
        assert "missing LLM config" in result.output
        process_input_mock.assert_not_called()

    def test_resets_active_display_before_cli_errors(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.wav"
        input_path.write_text("stub", encoding="utf-8")

        with (
            patch(
                "webinar_transcriber.cli.process_input",
                side_effect=OutputDirectoryExistsError("Output directory already exists: demo"),
            ),
            patch("webinar_transcriber.cli.StageReporter.reset_active_display") as reset_mock,
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
            patch("webinar_transcriber.cli.StageReporter", return_value=FakeReporter()),
        ):
            result = runner.invoke(main, [str(input_path)])

        assert result.exit_code == 130
        assert "Interrupted" in result.output
