"""Command line interface for webinar-transcriber."""

from pathlib import Path
from typing import IO, Any

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
    """Styled CLI error that keeps terminal failures visually consistent."""

    def show(self, file: IO[Any] | None = None) -> None:
        """Render the CLI error with consistent styling."""
        stream = file or click.get_text_stream("stderr")
        click.secho("Error:", fg="red", bold=True, nl=False, err=True, file=stream)
        click.echo(f" {self.format_message()}", err=True, file=stream)


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
    help="Override the whisper.cpp model path, for example "
    "'models/whisper-cpp/ggml-large-v3-turbo.bin'.",
)
@click.option(
    "--vad/--no-vad",
    default=True,
    show_default=True,
    help="Enable Silero speech-region detection before transcription planning.",
)
@click.option(
    "--vad-threshold",
    type=float,
    default=DEFAULT_VAD_THRESHOLD,
    show_default=True,
    help="Silero speech detection threshold.",
)
@click.option(
    "--min-speech-ms",
    type=int,
    default=DEFAULT_MIN_SPEECH_DURATION_MS,
    show_default=True,
    help="Minimum speech duration for Silero region detection.",
)
@click.option(
    "--min-silence-ms",
    type=int,
    default=DEFAULT_MIN_SILENCE_DURATION_MS,
    show_default=True,
    help="Minimum silence duration for Silero region separation.",
)
@click.option(
    "--speech-region-pad-ms",
    type=int,
    default=DEFAULT_SPEECH_REGION_PAD_MS,
    show_default=True,
    help="Symmetric extra context added around detected speech regions.",
)
@click.option(
    "--carryover/--no-carryover",
    default=True,
    show_default=True,
    help="Carry a bounded prompt suffix across adjacent inference windows.",
)
@click.option(
    "--carryover-max-sentences",
    type=int,
    default=DEFAULT_CARRYOVER_MAX_SENTENCES,
    show_default=True,
    help="Maximum trailing sentences to reuse as prompt carryover.",
)
@click.option(
    "--carryover-max-tokens",
    type=int,
    default=DEFAULT_CARRYOVER_MAX_TOKENS,
    show_default=True,
    help="Approximate token budget for prompt carryover per inference window.",
)
@click.option(
    "--threads",
    "asr_threads",
    type=int,
    default=None,
    show_default="auto",
    help="Number of whisper.cpp inference threads to use.",
)
@click.option(
    "--keep-audio/--no-keep-audio",
    default=False,
    show_default=True,
    help="Keep the normalized transcription audio as a run artifact.",
)
@click.option(
    "--audio-format",
    "kept_audio_format",
    type=click.Choice(["wav", "mp3"], case_sensitive=False),
    default="wav",
    show_default=True,
    help="Format for the kept transcription audio artifact.",
)
@click.option("--llm", is_flag=True, help="Enable optional provider-backed report enhancement.")
def main(
    input_path: Path,
    output_dir: Path | None,
    asr_model: str | None,
    vad: bool,
    vad_threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_region_pad_ms: int,
    carryover: bool,
    carryover_max_sentences: int,
    carryover_max_tokens: int,
    asr_threads: int | None,
    keep_audio: bool,
    kept_audio_format: str,
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
    asr_threads = asr_threads or default_asr_threads()

    try:
        process_input(
            input_path=input_path,
            output_dir=output_dir,
            asr_model=asr_model,
            vad=VadSettings(
                enabled=vad,
                threshold=vad_threshold,
                min_speech_duration_ms=min_speech_ms,
                min_silence_duration_ms=min_silence_ms,
                speech_region_pad_ms=speech_region_pad_ms,
            ),
            carryover=PromptCarryoverSettings(
                enabled=carryover,
                max_sentences=carryover_max_sentences,
                max_tokens=carryover_max_tokens,
            ),
            asr_threads=asr_threads,
            keep_audio=keep_audio,
            kept_audio_format=kept_audio_format,
            enable_llm=llm,
            reporter=reporter,
        )
    except KeyboardInterrupt:
        reporter.interrupted()
        raise click.exceptions.Exit(130) from None
    except (MediaProcessingError, OutputDirectoryExistsError) as error:
        reporter.reset_active_display()
        raise CLIError(str(error)) from error
