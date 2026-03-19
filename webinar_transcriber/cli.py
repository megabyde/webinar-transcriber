"""Command line interface for webinar-transcriber."""

from pathlib import Path

import click

from webinar_transcriber import __version__


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
    click.echo(
        "Bootstrap command ready for "
        f"{input_path} (ocr={ocr}, output_dir={output_dir}, format={output_format})."
    )
