"""Command line interface for webinar-transcriber."""

from __future__ import annotations

from pathlib import Path

import click
from click.core import ParameterSource

from webinar_transcriber import __version__
from webinar_transcriber.asr import (
    WHISPER_CPP_MODEL_FILENAME,
    AsrProcessingError,
    WhisperCppTranscriber,
    default_asr_threads,
)
from webinar_transcriber.diarization import DiarizationProcessingError, SherpaOnnxDiarizer
from webinar_transcriber.llm import (
    LlmConfigurationError,
    LlmProcessingError,
    build_llm_processor_from_env,
)
from webinar_transcriber.media import MediaProcessingError
from webinar_transcriber.paths import OutputDirectoryExistsError
from webinar_transcriber.processor import process_input, process_replay
from webinar_transcriber.replay import ReplayValidationError
from webinar_transcriber.ui import StageReporter


class CLIError(click.ClickException):
    """CLI error for actionable user-facing failures."""


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="webinar-transcriber")
@click.argument(
    "input_paths",
    nargs=-1,
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--from-run",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Rebuild reports from a completed run without media processing or transcription.",
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
    help="Number of local audio-processing threads. Defaults to the host CPU count, capped at 8.",
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
    help="Known exact speaker count to use for diarization. Omit for auto-clustering.",
)
@click.pass_context
def main(
    ctx: click.Context,
    input_paths: tuple[Path, ...],
    from_run: Path | None,
    output_dir: Path | None,
    asr_model: str | None,
    language: str | None,
    threads: int,
    keep_audio: bool,
    llm: bool,
    diarize: bool,
    diarize_speakers: int | None,
) -> None:
    """Transcribe INPUT files or rebuild one report with --from-run."""
    _validate_options(
        ctx,
        input_paths=input_paths,
        from_run=from_run,
        output_dir=output_dir,
        diarize=diarize,
        diarize_speakers=diarize_speakers,
    )

    reporter = StageReporter()

    try:
        llm_processor = build_llm_processor_from_env(threads=threads) if llm else None
        if from_run is not None:
            process_replay(
                source_run=from_run,
                output_dir=output_dir,
                llm_processor=llm_processor,
                reporter=reporter,
            )
            return
        for input_path in input_paths:
            diarizer = SherpaOnnxDiarizer(threads=threads) if diarize else None
            transcriber = WhisperCppTranscriber(
                model_name=asr_model, threads=threads, language=language
            )
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
    except KeyboardInterrupt:
        reporter.interrupted()
        raise click.exceptions.Exit(130) from None
    except (
        AsrProcessingError,
        DiarizationProcessingError,
        LlmConfigurationError,
        LlmProcessingError,
        MediaProcessingError,
        OutputDirectoryExistsError,
        ReplayValidationError,
    ) as ex:
        reporter.reset_active_display()
        raise CLIError(str(ex)) from ex


def _validate_options(
    ctx: click.Context,
    *,
    input_paths: tuple[Path, ...],
    from_run: Path | None,
    output_dir: Path | None,
    diarize: bool,
    diarize_speakers: int | None,
) -> None:
    if from_run is not None:
        if input_paths:
            raise CLIError("--from-run cannot be combined with input files.")
        replay_parameters = {
            "input_paths",
            "from_run",
            "output_dir",
            "threads",
            "llm",
        }
        incompatible = [
            parameter.name
            for parameter in ctx.command.params
            if parameter.name not in replay_parameters
            and ctx.get_parameter_source(parameter.name) is ParameterSource.COMMANDLINE
        ]
        if incompatible:
            formatted_options = ", ".join(
                f"--{option.replace('_', '-')}" for option in incompatible
            )
            raise CLIError(f"--from-run cannot be combined with {formatted_options}.")
        return
    if not input_paths:
        raise click.UsageError("Provide at least one INPUT or --from-run RUN_DIR.")
    if output_dir is not None and len(input_paths) > 1:
        raise CLIError("--output-dir can only be used with one input file.")
    if diarize_speakers is not None and not diarize:
        raise CLIError("--diarize-speakers requires --diarize.")
