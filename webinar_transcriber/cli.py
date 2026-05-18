"""Command line interface for webinar-transcriber."""

from __future__ import annotations

from pathlib import Path

import click

from webinar_transcriber import __version__
from webinar_transcriber.asr import (
    WHISPER_CPP_MODEL_FILENAME,
    ASRProcessingError,
    default_asr_threads,
)
from webinar_transcriber.diarization import DiarizationProcessingError
from webinar_transcriber.llm.contracts import LLMConfigurationError, LLMProcessingError
from webinar_transcriber.media import MediaProcessingError
from webinar_transcriber.paths import OutputDirectoryExistsError
from webinar_transcriber.processor import (
    DiarizationConfig,
    LLMConfig,
    TranscriptionConfig,
    process_input,
)
from webinar_transcriber.ui import RichStageReporter

_THREADS_DEFAULT = default_asr_threads()


class CLIError(click.ClickException):
    """CLI error for actionable user-facing failures."""


def _resolve_threads(_ctx: click.Context, _param: click.Parameter, value: int | None) -> int:
    if value is not None and value < 1:
        raise click.BadParameter("must be greater than or equal to 1")
    return value or default_asr_threads()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="webinar-transcriber")
@click.argument(
    "input_paths",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
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
    help="Override the whisper.cpp model identifier or local model path, for example "
    "'models/whisper-cpp/ggml-large-v3-turbo.bin'.",
)
@click.option(
    "--language",
    type=str,
    default=None,
    help="Force a Whisper language code hint, for example 'en' or 'ru'.",
)
@click.option(
    "--threads",
    type=int,
    metavar="INTEGER",
    default=_THREADS_DEFAULT,
    callback=_resolve_threads,
    show_default=True,
    help="Number of local audio-processing threads. Defaults to the host CPU count, capped at 8.",
)
@click.option(
    "--keep-audio",
    is_flag=True,
    help="Keep normalized transcription audio as mp3.",
)
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
def main(
    input_paths: tuple[Path, ...],
    output_dir: Path | None,
    asr_model: str | None,
    language: str | None,
    threads: int,
    keep_audio: bool,
    llm: bool,
    diarize: bool,
    diarize_speakers: int | None,
) -> None:
    """Transcribe one or more audio or video input files."""
    if output_dir is not None and len(input_paths) > 1:
        raise CLIError("--output-dir can only be used with one input file.")

    transcription_config = TranscriptionConfig(
        threads=threads,
        asr_model=asr_model,
        language=language,
        keep_audio=keep_audio,
    )
    llm_config = LLMConfig(processor="from_env" if llm else None)
    diarization_config = DiarizationConfig(
        enabled=diarize,
        speaker_count=diarize_speakers,
    )
    reporter = RichStageReporter()

    try:
        for input_path in input_paths:
            process_input(
                input_path=input_path,
                output_dir=output_dir,
                transcription_config=transcription_config,
                llm_config=llm_config,
                diarization_config=diarization_config,
                reporter=reporter,
            )
    except KeyboardInterrupt:
        reporter.interrupted()
        raise click.exceptions.Exit(130) from None
    except (
        ASRProcessingError,
        DiarizationProcessingError,
        LLMConfigurationError,
        LLMProcessingError,
        MediaProcessingError,
        OutputDirectoryExistsError,
    ) as error:
        reporter.reset_active_display()
        raise CLIError(str(error)) from error
