"""Command line interface for webinar-transcriber."""

import json
from pathlib import Path

import click

from webinar_transcriber import __version__
from webinar_transcriber.media import MediaProcessingError, probe_media
from webinar_transcriber.paths import OutputDirectoryExistsError, create_run_layout
from webinar_transcriber.processor import process_input
from webinar_transcriber.ui import RichStageReporter
from webinar_transcriber.video import (
    detect_scenes,
    estimate_sample_count,
    extract_representative_frames,
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="webinar-transcriber")
def main() -> None:
    """Run the webinar-transcriber CLI."""


@main.command()
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
    "--asr-backend",
    type=click.Choice(["auto", "faster-whisper", "mlx"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Select the ASR backend.",
)
@click.option(
    "--asr-model",
    type=str,
    default=None,
    help="Override the ASR model identifier, for example 'small' or an MLX repo name.",
)
def process(
    input_path: Path,
    output_dir: Path | None,
    output_format: str,
    asr_backend: str,
    asr_model: str | None,
) -> None:
    """Process an audio or video input file."""
    if not input_path.exists():
        raise click.ClickException(f"Input file does not exist: {input_path}")

    if not input_path.is_file():
        raise click.ClickException(f"Input path is not a file: {input_path}")

    try:
        artifacts = process_input(
            input_path=input_path,
            output_dir=output_dir,
            output_format=output_format,
            asr_backend=asr_backend,
            asr_model=asr_model,
            reporter=RichStageReporter(),
        )
    except (MediaProcessingError, OutputDirectoryExistsError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(f"Processed {input_path} into {artifacts.layout.run_dir} (format={output_format}).")


@main.command("extract-frames")
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
        raise click.ClickException(f"Input file does not exist: {input_path}")

    if not input_path.is_file():
        raise click.ClickException(f"Input path is not a file: {input_path}")

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
            raise click.ClickException("Frame extraction is only supported for video input.")

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
    except (MediaProcessingError, OutputDirectoryExistsError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(f"Extracted {len(frames)} frames into {layout.run_dir}.")
