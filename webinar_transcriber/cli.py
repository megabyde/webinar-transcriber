"""Command line interface for webinar-transcriber."""

from __future__ import annotations

from pathlib import Path

import click

from webinar_transcriber import __version__
from webinar_transcriber.asr import (
    WHISPER_CPP_MODEL_FILENAME,
    AsrConfigurationError,
    AsrProcessingError,
    WhisperCppTranscriber,
    default_asr_threads,
)
from webinar_transcriber.diarization import (
    DiarizationConfigurationError,
    DiarizationProcessingError,
    SherpaOnnxDiarizer,
)
from webinar_transcriber.llm import (
    LlmConfigurationError,
    LlmProcessingError,
    build_llm_processor_from_env,
)
from webinar_transcriber.llm.rerun import (
    LlmRerunError,
    load_llm_rerun_source,
    rerun_llm_report,
)
from webinar_transcriber.media import MediaProcessingError
from webinar_transcriber.paths import OutputDirectoryExistsError
from webinar_transcriber.processor import process_input
from webinar_transcriber.ui import StageReporter


class CLIError(click.ClickException):
    """CLI error for actionable user-facing failures."""


# Failures of the model, host, or provider setup rather than of one file. Every input would hit
# them identically, so the first one stops the batch.
SETUP_ERRORS = (
    AsrConfigurationError,
    DiarizationConfigurationError,
    LlmConfigurationError,
)
# Failures that belong to one input. A batch reports them and moves on to the next file. These are
# the base classes of the setup errors above, so the handlers must stay in this order.
INPUT_ERRORS = (
    AsrProcessingError,
    DiarizationProcessingError,
    LlmProcessingError,
    MediaProcessingError,
    OutputDirectoryExistsError,
)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="webinar-transcriber")
@click.argument(
    "input_paths",
    nargs=-1,
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Media files to transcribe.",
)
@click.option(
    "--rerun-llm",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    metavar="RUN_DIR",
    help="Regenerate LLM-polished reports from a completed run.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Write artifacts to a specific output directory.",
)
@click.option(
    "--asr-model",
    type=str,
    default=WHISPER_CPP_MODEL_FILENAME,
    show_default=True,
    help=(
        "Override the whisper.cpp model identifier or local model path, for example "
        "'models/whisper-cpp/ggml-large-v3-turbo.bin'."
    ),
)
@click.option(
    "--language",
    type=str,
    default=None,
    help="Force a Whisper language code hint, for example 'en' or 'ru'.",
)
@click.option(
    "--threads",
    type=click.IntRange(min=1),
    metavar="INTEGER",
    default=default_asr_threads(),
    show_default=True,
    help=(
        "Number of local processing and concurrent LLM threads. "
        "Defaults to the host CPU count, capped at 8."
    ),
)
@click.option("--keep-audio", is_flag=True, help="Keep normalized transcription audio as mp3.")
@click.option("--llm", is_flag=True, help="Enable optional provider-backed report enhancement.")
@click.option(
    "--diarize/--no-diarize",
    default=False,
    show_default=True,
    help="Enable local speaker diarization.",
)
@click.option(
    "--diarize-speakers",
    type=click.IntRange(min=1, max=20),
    default=None,
    metavar="COUNT",
    help="Known speaker count; surplus speakers are folded in. Omit for auto-clustering.",
)
def main(
    input_paths: tuple[Path, ...],
    rerun_llm: Path | None,
    output_dir: Path | None,
    asr_model: str | None,
    language: str | None,
    threads: int,
    keep_audio: bool,
    llm: bool,
    diarize: bool,
    diarize_speakers: int | None,
) -> None:
    """Transcribe media inputs or regenerate a completed run's LLM reports."""
    _validate_cli_args(
        input_paths,
        rerun_llm=rerun_llm,
        output_dir=output_dir,
        diarize=diarize,
        diarize_speakers=diarize_speakers,
    )

    reporter = StageReporter()

    if rerun_llm is not None:
        _run_llm_rerun(rerun_llm, threads=threads, reporter=reporter)
        return

    # A provider misconfiguration would fail every input identically, so it aborts before any run
    try:
        llm_processor = build_llm_processor_from_env(threads=threads) if llm else None
    except LlmConfigurationError as ex:
        raise CLIError(str(ex)) from ex

    failed_paths: list[Path] = []
    try:
        for input_path in input_paths:
            diarizer = SherpaOnnxDiarizer(threads=threads) if diarize else None
            transcriber = WhisperCppTranscriber(
                model_name=asr_model, threads=threads, language=language
            )
            try:
                process_input(
                    input_path=input_path,
                    output_dir=output_dir,
                    threads=threads,
                    keep_audio=keep_audio,
                    llm_processor=llm_processor,
                    diarizer=diarizer,
                    diarization_speaker_count=diarize_speakers,
                    transcriber=transcriber,
                    reporter=reporter,
                )
            except SETUP_ERRORS as ex:
                reporter.reset_active_display()
                raise CLIError(str(ex)) from ex
            except INPUT_ERRORS as ex:
                failed_paths.append(input_path)
                reporter.failed_run(input_path, str(ex))
    except KeyboardInterrupt:
        reporter.interrupted()
        raise click.exceptions.Exit(130) from None

    if failed_paths:
        # One input already printed its own failure line; a batch needs the tally too
        if len(input_paths) > 1:
            reporter.batch_summary(
                succeeded=len(input_paths) - len(failed_paths), failed=len(failed_paths)
            )
        raise click.exceptions.Exit(1)


def _validate_cli_args(
    input_paths: tuple[Path, ...],
    *,
    rerun_llm: Path | None,
    output_dir: Path | None,
    diarize: bool,
    diarize_speakers: int | None,
) -> None:
    if not input_paths and rerun_llm is None:
        raise CLIError("Provide at least one input file or --rerun-llm RUN_DIR.")
    if rerun_llm is not None:
        if input_paths:
            raise CLIError("--rerun-llm cannot be used with input files.")
        if output_dir is not None:
            raise CLIError("--rerun-llm cannot be used with --output-dir.")
        return
    if output_dir is not None and len(input_paths) > 1:
        raise CLIError("--output-dir can only be used with one input file.")
    if diarize_speakers is not None and not diarize:
        raise CLIError("--diarize-speakers requires --diarize.")


def _run_llm_rerun(run_dir: Path, *, threads: int, reporter: StageReporter) -> None:
    try:
        source = load_llm_rerun_source(run_dir)
        llm_processor = build_llm_processor_from_env(threads=threads)
        rerun_llm_report(source, llm_processor=llm_processor, reporter=reporter)
    except LlmConfigurationError as ex:
        raise CLIError(str(ex)) from ex
    except LlmRerunError as ex:
        reporter.reset_active_display()
        raise CLIError(str(ex)) from ex
    except KeyboardInterrupt:
        reporter.interrupted()
        raise click.exceptions.Exit(130) from None
