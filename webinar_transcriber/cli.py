"""Command line interface for webinar-transcriber."""

from pathlib import Path

import click

from webinar_transcriber import __version__
from webinar_transcriber.media import MediaProcessingError
from webinar_transcriber.paths import OutputDirectoryExistsError
from webinar_transcriber.processor import process_input
from webinar_transcriber.ui import RichStageReporter


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="webinar-transcriber")
def main() -> None:
    """Run the webinar-transcriber CLI."""


@main.command()
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option("--ocr", is_flag=True, help="Enable OCR-assisted slide alignment for video input.")
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
def process(input_path: Path, ocr: bool, output_dir: Path | None, output_format: str) -> None:
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
            ocr_enabled=ocr,
            reporter=RichStageReporter(),
        )
    except (MediaProcessingError, OutputDirectoryExistsError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(
        "Processed "
        f"{input_path} into {artifacts.layout.run_dir} "
        f"(ocr={artifacts.report.ocr_enabled}, format={output_format})."
    )
