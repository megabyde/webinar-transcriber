"""Command line interface for webinar-transcriber."""

from __future__ import annotations

from pathlib import Path

import click

from webinar_transcriber import __version__
from webinar_transcriber.asr import (
    DEFAULT_CARRYOVER_MAX_SENTENCES,
    DEFAULT_CARRYOVER_MAX_TOKENS,
    PromptCarryoverSettings,
    default_asr_threads,
)
from webinar_transcriber.media import MediaProcessingError
from webinar_transcriber.paths import OutputDirectoryExistsError
from webinar_transcriber.processor import process_input
from webinar_transcriber.segmentation import (
    DEFAULT_MIN_SILENCE_DURATION_MS,
    DEFAULT_MIN_SPEECH_DURATION_MS,
    DEFAULT_SPEECH_REGION_PAD_MS,
    DEFAULT_VAD_THRESHOLD,
    VadSettings,
)
from webinar_transcriber.ui import RichStageReporter


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
    "--vad/--no-vad",
    default=True,
    show_default=True,
    help="Enable Silero speech-region detection before transcription planning.",
)
@click.option(
    "--carryover/--no-carryover",
    default=True,
    show_default=True,
    help="Carry a bounded prompt suffix across adjacent inference windows.",
)
@click.option(
    "--keep-audio",
    "kept_audio_format",
    type=click.Choice(["wav", "mp3"], case_sensitive=False),
    default=None,
    metavar="FORMAT",
    help="Keep normalized transcription audio as FORMAT: wav or mp3.",
)
@click.option("--llm", is_flag=True, help="Enable optional provider-backed report enhancement.")
def main(
    input_path: Path,
    output_dir: Path | None,
    asr_model: str | None,
    language: str | None,
    vad: bool,
    carryover: bool,
    kept_audio_format: str | None,
    llm: bool,
) -> None:
    """Transcribe an audio or video input file.

    Raises:
        CLIError: If the input path is missing, invalid, or media setup fails.
        click.exceptions.Exit: If the user interrupts the run.
    """
    if not input_path.exists():
        raise CLIError(f"Input file does not exist: {input_path}")

    if not input_path.is_file():
        raise CLIError(f"Input path is not a file: {input_path}")

    reporter = RichStageReporter()

    try:
        process_input(
            input_path=input_path,
            output_dir=output_dir,
            asr_model=asr_model,
            language=language,
            vad=VadSettings(
                enabled=vad,
                threshold=DEFAULT_VAD_THRESHOLD,
                min_speech_duration_ms=DEFAULT_MIN_SPEECH_DURATION_MS,
                min_silence_duration_ms=DEFAULT_MIN_SILENCE_DURATION_MS,
                speech_region_pad_ms=DEFAULT_SPEECH_REGION_PAD_MS,
            ),
            carryover=PromptCarryoverSettings(
                enabled=carryover,
                max_sentences=DEFAULT_CARRYOVER_MAX_SENTENCES,
                max_tokens=DEFAULT_CARRYOVER_MAX_TOKENS,
            ),
            asr_threads=default_asr_threads(),
            keep_audio=kept_audio_format is not None,
            kept_audio_format=kept_audio_format or "wav",
            enable_llm=llm,
            reporter=reporter,
        )
    except KeyboardInterrupt:
        reporter.interrupted()
        raise click.exceptions.Exit(130) from None
    except (MediaProcessingError, OutputDirectoryExistsError) as error:
        reporter.reset_active_display()
        raise CLIError(str(error)) from error
