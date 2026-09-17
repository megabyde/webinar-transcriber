"""Tests for the CLI entrypoints."""

import runpy
import sys
from unittest.mock import ANY, patch

import pytest
from click.testing import CliRunner

from webinar_transcriber import __version__
from webinar_transcriber.asr import (
    AsrConfigurationError,
    AsrProcessingError,
    default_asr_threads,
)
from webinar_transcriber.cli import main
from webinar_transcriber.diarization import DiarizationConfigurationError
from webinar_transcriber.llm import LlmConfigurationError, LlmProcessingError
from webinar_transcriber.llm.rerun import LlmRerunError
from webinar_transcriber.media import MediaProcessingError
from webinar_transcriber.paths import OutputDirectoryExistsError
from webinar_transcriber.tests.conftest import process_artifacts


class TestCli:
    def test_main_help_describes_root_command(self) -> None:
        runner = CliRunner()

        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Positional arguments:" in result.output
        assert "[INPUT_PATHS]...  Media files to transcribe." in result.output
        assert "Transcribe media inputs or regenerate a completed run's LLM reports." in (
            result.output
        )

    def test_main_version_prints_package_version(self) -> None:
        runner = CliRunner()

        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert result.output == f"webinar-transcriber, version {__version__}\n"

    def test_module_entrypoint_reports_version(self) -> None:
        original_argv = sys.argv[:]
        sys.argv = ["python", "--version"]

        try:
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
        assert "--rerun-llm" in result.output
        assert "--diarize / --no-diarize" in result.output
        assert "--diarize-speakers" in result.output
        assert "Override the whisper.cpp model identifier" in result.output
        assert "[default: large-v3-turbo]" in result.output
        assert "Force a Whisper language code hint" in result.output
        assert "model path" in result.output
        assert "provider-backed report" in result.output
        assert "enhancement." in result.output

    def test_requires_input_or_llm_rerun(self) -> None:
        result = CliRunner().invoke(main)

        assert result.exit_code != 0
        assert "Provide at least one input file or --rerun-llm RUN_DIR" in result.output

    def test_reruns_llm_without_starting_media_pipeline(self, tmp_path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        source = object()
        llm_processor = object()

        with (
            patch(
                "webinar_transcriber.cli.load_llm_rerun_source", return_value=source
            ) as load_source_mock,
            patch(
                "webinar_transcriber.cli.build_llm_processor_from_env",
                return_value=llm_processor,
            ) as build_llm_processor_mock,
            patch("webinar_transcriber.cli.rerun_llm_report") as rerun_mock,
            patch("webinar_transcriber.cli.process_input") as process_input_mock,
        ):
            result = CliRunner().invoke(
                main,
                [
                    "--rerun-llm",
                    str(run_dir),
                    "--threads",
                    "3",
                    "--asr-model",
                    "unused.bin",
                    "--language",
                    "ru",
                    "--keep-audio",
                    "--llm",
                    "--diarize-speakers",
                    "2",
                ],
            )

        assert result.exit_code == 0
        load_source_mock.assert_called_once_with(run_dir)
        build_llm_processor_mock.assert_called_once_with(threads=3)
        rerun_mock.assert_called_once_with(source, llm_processor=llm_processor, reporter=ANY)
        process_input_mock.assert_not_called()

    def test_rejects_llm_rerun_with_input(self, tmp_path) -> None:
        input_path = tmp_path / "demo.wav"
        input_path.write_text("stub", encoding="utf-8")
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        result = CliRunner().invoke(main, [str(input_path), "--rerun-llm", str(run_dir)])

        assert result.exit_code != 0
        assert "--rerun-llm cannot be used with input files" in result.output

    def test_rejects_llm_rerun_with_output_directory(self, tmp_path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        result = CliRunner().invoke(
            main, ["--rerun-llm", str(run_dir), "--output-dir", str(tmp_path / "output")]
        )

        assert result.exit_code != 0
        assert "--rerun-llm cannot be used with --output-dir" in result.output

    def test_reports_llm_configuration_error_during_rerun(self, tmp_path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        with (
            patch("webinar_transcriber.cli.load_llm_rerun_source", return_value=object()),
            patch(
                "webinar_transcriber.cli.build_llm_processor_from_env",
                side_effect=LlmConfigurationError("missing LLM config"),
            ),
        ):
            result = CliRunner().invoke(main, ["--rerun-llm", str(run_dir)])

        assert result.exit_code != 0
        assert "missing LLM config" in result.output

    def test_reports_invalid_llm_rerun_source(self, tmp_path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        with (
            patch(
                "webinar_transcriber.cli.load_llm_rerun_source",
                side_effect=LlmRerunError("unsafe source run"),
            ),
            patch("webinar_transcriber.cli.build_llm_processor_from_env") as build_mock,
        ):
            result = CliRunner().invoke(main, ["--rerun-llm", str(run_dir)])

        assert result.exit_code != 0
        assert "unsafe source run" in result.output
        build_mock.assert_not_called()

    def test_handles_ctrl_c_during_llm_rerun(self, tmp_path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        with (
            patch("webinar_transcriber.cli.load_llm_rerun_source", return_value=object()),
            patch("webinar_transcriber.cli.build_llm_processor_from_env", return_value=object()),
            patch("webinar_transcriber.cli.rerun_llm_report", side_effect=KeyboardInterrupt),
        ):
            result = CliRunner().invoke(main, ["--rerun-llm", str(run_dir)])

        assert result.exit_code == 130

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
            (LlmProcessingError("LLM request failed"), "LLM request failed"),
        ],
    )
    def test_reports_expected_runtime_errors(
        self, tmp_path, error: Exception, message: str
    ) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.wav"
        input_path.write_text("stub", encoding="utf-8")

        with patch("webinar_transcriber.cli.process_input", side_effect=error):
            result = runner.invoke(main, [str(input_path)])

        assert result.exit_code != 0
        assert message in result.output

    def test_continues_the_batch_past_a_failing_input(self, tmp_path) -> None:
        runner = CliRunner()
        paths = []
        for name in ("first", "second", "third"):
            path = tmp_path / f"{name}.mp4"
            path.write_text("stub", encoding="utf-8")
            paths.append(path)

        def run(*, input_path, **_kwargs):
            if input_path == paths[1]:
                raise MediaProcessingError(f"No audio stream found in {input_path.name}.")
            return process_artifacts(input_path, tmp_path / f"run-{input_path.stem}")

        with patch("webinar_transcriber.cli.process_input", side_effect=run) as process_input_mock:
            result = runner.invoke(main, [str(path) for path in paths])

        assert [call.kwargs["input_path"] for call in process_input_mock.call_args_list] == paths
        assert "No audio stream found in second.mp4." in result.output
        assert "2 succeeded, 1 failed" in result.output
        assert result.exit_code == 1

    @pytest.mark.parametrize(
        ("error", "message"),
        [
            (AsrConfigurationError("whisper.cpp model file does not exist"), "does not exist"),
            (
                DiarizationConfigurationError("sherpa-onnx is unavailable"),
                "sherpa-onnx is unavailable",
            ),
        ],
        ids=["asr-model", "diarization-runtime"],
    )
    def test_shared_setup_failure_stops_the_batch_after_one_attempt(
        self, tmp_path, error: Exception, message: str
    ) -> None:
        runner = CliRunner()
        paths = []
        for name in ("first", "second", "third"):
            path = tmp_path / f"{name}.mp4"
            path.write_text("stub", encoding="utf-8")
            paths.append(path)

        with patch(
            "webinar_transcriber.cli.process_input", side_effect=error
        ) as process_input_mock:
            result = runner.invoke(main, [str(path) for path in paths])

        assert process_input_mock.call_count == 1
        assert message in result.output
        assert "succeeded" not in result.output
        assert result.exit_code != 0

    def test_single_failing_input_reports_no_batch_tally(self, tmp_path) -> None:
        runner = CliRunner()
        input_path = tmp_path / "demo.wav"
        input_path.write_text("stub", encoding="utf-8")

        with patch(
            "webinar_transcriber.cli.process_input",
            side_effect=MediaProcessingError("No audio stream found in demo.wav."),
        ):
            result = runner.invoke(main, [str(input_path)])

        assert "No audio stream found in demo.wav." in result.output
        assert "succeeded" not in result.output
        assert result.exit_code == 1

    def test_ctrl_c_abandons_the_rest_of_the_batch(self, tmp_path) -> None:
        runner = CliRunner()
        first_input = tmp_path / "first.mp4"
        second_input = tmp_path / "second.mp4"
        first_input.write_text("stub", encoding="utf-8")
        second_input.write_text("stub", encoding="utf-8")

        with patch(
            "webinar_transcriber.cli.process_input", side_effect=KeyboardInterrupt
        ) as process_input_mock:
            result = runner.invoke(main, [str(first_input), str(second_input)])

        assert process_input_mock.call_count == 1
        assert result.exit_code == 130

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
