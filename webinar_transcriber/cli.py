"""Command line interface for webinar-transcriber."""

import json
from pathlib import Path

import click

from webinar_transcriber import __version__
from webinar_transcriber.asr import DEFAULT_ASR_THREADS
from webinar_transcriber.media import MediaProcessingError, probe_media
from webinar_transcriber.paths import OutputDirectoryExistsError, create_run_layout
from webinar_transcriber.processor import process_input
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
    help="Enable speech-region detection before chunked transcription.",
)
@click.option(
    "--chunk-target-sec",
    type=float,
    default=20.0,
    show_default=True,
    help="Target chunk duration for ASR planning.",
)
@click.option(
    "--chunk-max-sec",
    type=float,
    default=30.0,
    show_default=True,
    help="Hard cap for chunk duration.",
)
@click.option(
    "--chunk-overlap-sec",
    type=float,
    default=1.5,
    show_default=True,
    help="Overlap between adjacent ASR chunks.",
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
    chunk_target_sec: float,
    chunk_max_sec: float,
    chunk_overlap_sec: float,
    asr_threads: int,
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
            chunk_target_sec=chunk_target_sec,
            chunk_max_sec=chunk_max_sec,
            chunk_overlap_sec=chunk_overlap_sec,
            asr_threads=asr_threads,
            enable_llm=llm,
            reporter=reporter,
        )
    except KeyboardInterrupt:
        reporter.interrupted()
        raise click.exceptions.Exit(130) from None
    except (MediaProcessingError, OutputDirectoryExistsError) as error:
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
        if str(media_asset.media_type) != "video":
            raise CLIError("Frame extraction is only supported for video input.")

        reporter.progress_started(
            "detect_scenes",
            "Detecting scenes",
            total=estimate_sample_count(media_asset.duration_sec),
        )
        scenes = detect_scenes(
            input_path,
            duration_sec=media_asset.duration_sec,
            progress_callback=lambda: reporter.progress_advanced("detect_scenes"),
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
        raise CLIError(str(error)) from error

    click.echo(f"Extracted {len(frames)} frames into {layout.run_dir}.")
