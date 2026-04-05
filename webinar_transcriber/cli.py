"""Command line interface for webinar-transcriber."""

import json
from pathlib import Path

import click

from webinar_transcriber import __version__
from webinar_transcriber.asr import (
    DEFAULT_ASR_THREADS,
    DEFAULT_CARRYOVER_MAX_SENTENCES,
    DEFAULT_CARRYOVER_MAX_TOKENS,
)
from webinar_transcriber.media import MediaProcessingError, probe_media
from webinar_transcriber.models import MediaType
from webinar_transcriber.paths import OutputDirectoryExistsError, create_run_layout
from webinar_transcriber.processor import process_input
from webinar_transcriber.segmentation import (
    DEFAULT_MIN_SILENCE_DURATION_MS,
    DEFAULT_MIN_SPEECH_DURATION_MS,
    DEFAULT_SPEECH_REGION_PAD_MS,
    DEFAULT_VAD_THRESHOLD,
)
from webinar_transcriber.ui import RichStageReporter
from webinar_transcriber.video import (
    detect_scenes,
    estimate_sample_count,
    extract_representative_frames,
)


class CLIError(click.ClickException):
    """Styled CLI error that keeps terminal failures visually consistent."""

    def show(self, file=None) -> None:
        stream = file or click.get_text_stream("stderr")
        click.secho("Error:", fg="red", bold=True, nl=False, err=True, file=stream)
        click.echo(f" {self.format_message()}", err=True, file=stream)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="webinar-transcriber")
def main() -> None:
    """Run the webinar-transcriber CLI."""


@main.command(short_help="Process an audio or video input.")
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Write artifacts to a specific output directory.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["all", "md", "docx", "json"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Select which report format to write.",
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
    default=DEFAULT_ASR_THREADS,
    show_default=True,
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
@click.option(
    "--llm",
    is_flag=True,
    help="Enable optional OpenAI-based report enhancement.",
)
def process(
    input_path: Path,
    output_dir: Path | None,
    output_format: str,
    asr_model: str | None,
    vad: bool,
    vad_threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_region_pad_ms: int,
    carryover: bool,
    carryover_max_sentences: int,
    carryover_max_tokens: int,
    asr_threads: int,
    keep_audio: bool,
    kept_audio_format: str,
    llm: bool,
) -> None:
    """Process an audio or video input file."""
    if not input_path.exists():
        raise CLIError(f"Input file does not exist: {input_path}")

    if not input_path.is_file():
        raise CLIError(f"Input path is not a file: {input_path}")

    reporter = RichStageReporter()

    try:
        process_input(
            input_path=input_path,
            output_dir=output_dir,
            output_format=output_format,
            asr_model=asr_model,
            vad_enabled=vad,
            vad_threshold=vad_threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_region_pad_ms=speech_region_pad_ms,
            carryover_enabled=carryover,
            carryover_max_sentences=carryover_max_sentences,
            carryover_max_tokens=carryover_max_tokens,
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


@main.command("extract-frames", short_help="Extract representative frames from a video.")
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Write extracted frames to a specific output directory.",
)
def extract_frames(input_path: Path, output_dir: Path | None) -> None:
    """Detect scenes and extract representative frames from a video input."""
    if not input_path.exists():
        raise CLIError(f"Input file does not exist: {input_path}")

    if not input_path.is_file():
        raise CLIError(f"Input path is not a file: {input_path}")

    reporter = RichStageReporter()

    try:
        reporter.begin_run(input_path, output_format="frames")
        reporter.stage_started("prepare_run_dir", "Preparing run directory")
        layout = create_run_layout(input_path=input_path, output_dir=output_dir)
        reporter.stage_finished(
            "prepare_run_dir", "Preparing run directory", detail=str(layout.run_dir)
        )

        reporter.stage_started("probe_media", "Probing media")
        media_asset = probe_media(input_path)
        reporter.stage_finished(
            "probe_media",
            "Probing media",
            detail=f"{media_asset.media_type.value}, {media_asset.duration_sec:.1f}s",
        )
        if media_asset.media_type != MediaType.VIDEO:
            raise CLIError("Frame extraction is only supported for video input.")

        reporter.progress_started(
            "detect_scenes",
            "Detecting scenes",
            total=estimate_sample_count(media_asset.duration_sec),
            count_label="s",
            detail="0 scenes",
        )
        scenes = detect_scenes(
            input_path,
            duration_sec=media_asset.duration_sec,
            progress_callback=lambda scene_count: reporter.progress_advanced(
                "detect_scenes",
                detail=f"{scene_count} {'scene' if scene_count == 1 else 'scenes'}",
            ),
        )
        reporter.stage_finished("detect_scenes", "Detecting scenes", detail=f"{len(scenes)} scenes")

        reporter.progress_started(
            "extract_frames",
            "Extracting slide frames",
            total=len(scenes),
        )
        frames = extract_representative_frames(
            input_path,
            scenes,
            layout.frames_dir,
            progress_callback=lambda: reporter.progress_advanced("extract_frames"),
        )
        reporter.stage_finished(
            "extract_frames",
            "Extracting slide frames",
            detail=f"{len(frames)} frames",
        )

        layout.scenes_path.write_text(
            json.dumps({"scenes": [scene.model_dump(mode="json") for scene in scenes]}, indent=2),
            encoding="utf-8",
        )
    except KeyboardInterrupt:
        reporter.interrupted()
        raise click.exceptions.Exit(130) from None
    except (MediaProcessingError, OutputDirectoryExistsError) as error:
        reporter.reset_active_display()
        raise CLIError(str(error)) from error

    click.echo(f"Extracted {len(frames)} frames into {layout.run_dir}.")
