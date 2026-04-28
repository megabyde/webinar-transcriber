"""Helpers for deterministic run-directory construction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


class OutputDirectoryExistsError(FileExistsError):
    """Raised when a run output directory already exists."""


def _slugify_stem(path: Path) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", path.stem).strip("-").lower()
    return stem or "input"


@dataclass(frozen=True)
class RunLayout:
    """Computed paths for a single processing run."""

    run_dir: Path

    @property
    def metadata_path(self) -> Path:
        """Return the metadata artifact path."""
        return self.run_dir / "metadata.json"

    @property
    def transcript_path(self) -> Path:
        """Return the transcript JSON artifact path."""
        return self.run_dir / "transcript.json"

    @property
    def subtitle_vtt_path(self) -> Path:
        """Return the subtitle VTT artifact path."""
        return self.run_dir / "transcript.vtt"

    @property
    def scenes_path(self) -> Path:
        """Return the scene metadata artifact path."""
        return self.run_dir / "scenes.json"

    @property
    def diagnostics_path(self) -> Path:
        """Return the diagnostics artifact path."""
        return self.run_dir / "diagnostics.json"

    @property
    def asr_dir(self) -> Path:
        """Return the ASR intermediate artifact directory."""
        return self.run_dir / "asr"

    @property
    def speech_regions_path(self) -> Path:
        """Return the speech-region artifact path."""
        return self.asr_dir / "speech_regions.json"

    @property
    def decoded_windows_path(self) -> Path:
        """Return the decoded-window artifact path."""
        return self.asr_dir / "decoded_windows.json"

    @property
    def markdown_report_path(self) -> Path:
        """Return the Markdown report path."""
        return self.run_dir / "report.md"

    @property
    def docx_report_path(self) -> Path:
        """Return the DOCX report path."""
        return self.run_dir / "report.docx"

    @property
    def json_report_path(self) -> Path:
        """Return the JSON report path."""
        return self.run_dir / "report.json"

    def transcription_audio_path(self, audio_format: str = "wav") -> Path:
        """Return the kept transcription-audio artifact path."""
        return self.run_dir / f"transcription-audio.{audio_format}"

    @property
    def frames_dir(self) -> Path:
        """Return the extracted-frame artifact directory."""
        return self.run_dir / "frames"


def build_run_layout(
    input_path: Path, output_dir: Path | None = None, *, now: datetime | None = None
) -> RunLayout:
    """Return the run layout without touching the filesystem.

    Returns:
        RunLayout: The computed run layout.

    Raises:
        OutputDirectoryExistsError: If the target run directory already exists.
    """
    if output_dir is None:
        current_time = now or datetime.now(tz=UTC).astimezone()
        timestamp = current_time.strftime("%Y%m%d-%H%M%S-%f")
        run_dir = Path("runs") / f"{timestamp}_{_slugify_stem(input_path)}"
    else:
        run_dir = output_dir

    if run_dir.exists():
        raise OutputDirectoryExistsError(f"Output directory already exists: {run_dir}")

    return RunLayout(run_dir=run_dir)


def create_run_layout(
    input_path: Path, output_dir: Path | None = None, *, now: datetime | None = None
) -> RunLayout:
    """Create the run directory and return its layout.

    Returns:
        RunLayout: The created run layout.
    """
    layout = build_run_layout(input_path=input_path, output_dir=output_dir, now=now)
    layout.run_dir.mkdir(parents=True)
    return layout
