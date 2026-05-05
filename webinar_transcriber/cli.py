"""Command line interface for webinar-transcriber."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import click

from webinar_transcriber import __version__
from webinar_transcriber.asr import (
    ASRProcessingError,
    PromptCarryoverSettings,
    default_asr_threads,
)
from webinar_transcriber.llm.contracts import LLMConfigurationError, LLMProcessingError
from webinar_transcriber.media import MediaProcessingError
from webinar_transcriber.paths import OutputDirectoryExistsError
from webinar_transcriber.processor import process_input
from webinar_transcriber.segmentation import (
    VadSettings,
)
from webinar_transcriber.ui import RichStageReporter

if TYPE_CHECKING:
    from webinar_transcriber.normalized_audio import TranscriptionAudioFormat


class CLIError(click.ClickException):
    """CLI error for actionable user-facing failures."""


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="webinar-transcriber")
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Write artifacts to a specific output directory.",
)
@click.option(
    "--asr-model",
    type=str,
    default=None,
    help="Override the whisper.cpp model identifier or local model path, for example "
    "'large-v3-turbo' or 'models/whisper-cpp/ggml-large-v3-turbo.bin'.",
)
@click.option(
    "--language",
    type=str,
    default=None,
    help="Force a Whisper language code, for example 'en' or 'ru'.",
)
@click.option(
    "--threads",
    type=click.IntRange(min=1),
    default=None,
    help="Number of ASR threads. Defaults to the host CPU count.",
)
@click.option(
    "--vad/--no-vad",
    default=True,
    show_default=True,
    help="Enable Silero speech-region detection before transcription planning.",
)
@click.option(
    "--keep-audio",
    type=click.Choice(["wav", "mp3"], case_sensitive=False),
    default=None,
    flag_value="mp3",
    is_flag=False,
    metavar="[FORMAT]",
    help="Keep normalized transcription audio as FORMAT, defaulting to mp3.",
)
@click.option("--llm", is_flag=True, help="Enable optional provider-backed report enhancement.")
def main(
    input_path: Path,
    output_dir: Path | None,
    asr_model: str | None,
    language: str | None,
    threads: int | None,
    vad: bool,
    keep_audio: str | None,
    llm: bool,
) -> None:
    """Transcribe an audio or video input file."""
    if not input_path.exists():
        raise CLIError(f"Input file does not exist: {input_path}")

    if not input_path.is_file():
        raise CLIError(f"Input path is not a file: {input_path}")

    reporter = RichStageReporter()
    kept_format = cast(
        "TranscriptionAudioFormat | None", keep_audio.lower() if keep_audio else None
    )

    try:
        process_input(
            input_path=input_path,
            output_dir=output_dir,
            asr_model=asr_model,
            language=language,
            vad=VadSettings(enabled=vad),
            carryover=PromptCarryoverSettings(),
            asr_threads=threads or default_asr_threads(),
            keep_audio=kept_format,
            enable_llm=llm,
            reporter=reporter,
        )
    except KeyboardInterrupt:
        reporter.interrupted()
        raise click.exceptions.Exit(130) from None
    except (
        ASRProcessingError,
        LLMConfigurationError,
        LLMProcessingError,
        MediaProcessingError,
        OutputDirectoryExistsError,
    ) as error:
        reporter.reset_active_display()
        raise CLIError(str(error)) from error
