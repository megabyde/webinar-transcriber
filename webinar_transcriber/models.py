"""Typed models used across the processing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence


_COMPACT_OPTIONAL_KEYS = frozenset({"speaker", "scene_id", "frame_id"})


def _compact_dict_factory(items: list[tuple[str, object]]) -> dict[str, object]:
    return {k: v for k, v in items if not (k in _COMPACT_OPTIONAL_KEYS and v is None)}


# ---------------------------------------------------------------------------
# Timeline primitives
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class TimelineSpan:
    """Timeline-bounded model."""

    start_sec: float
    end_sec: float

    @property
    def midpoint(self) -> float:
        """Return the midpoint on the timeline."""
        return (self.start_sec + self.end_sec) / 2.0

    @property
    def duration_sec(self) -> float:
        """Return the non-negative duration of the item."""
        return max(0.0, self.end_sec - self.start_sec)

    def gap_before(self, other: TimelineSpan) -> float:
        """Return the non-negative gap before another timeline item."""
        return max(0.0, other.start_sec - self.end_sec)


@dataclass(slots=True, frozen=True)
class TimelineItem(TimelineSpan):
    """Timeline-bounded model with a stable identifier."""

    id: str


def average_duration_sec(items: Sequence[TimelineSpan]) -> float | None:
    """Return the average duration for timeline-bounded items."""
    return sum(item.duration_sec for item in items) / len(items) if items else None


# ---------------------------------------------------------------------------
# Media assets
# ---------------------------------------------------------------------------


class MediaType(StrEnum):
    """Supported top-level media types."""

    AUDIO = "audio"
    VIDEO = "video"


@dataclass(slots=True, frozen=True)
class BaseMediaAsset:
    """Shared metadata for a probed media asset."""

    path: str
    duration_sec: float
    sample_rate: int | None = None
    channels: int | None = None


@dataclass(slots=True, frozen=True)
class AudioAsset(BaseMediaAsset):
    """Metadata for a probed audio-only asset."""

    media_type: Literal[MediaType.AUDIO] = MediaType.AUDIO


@dataclass(slots=True, frozen=True)
class VideoAsset(BaseMediaAsset):
    """Metadata for a probed video asset."""

    media_type: Literal[MediaType.VIDEO] = MediaType.VIDEO
    fps: float | None = None
    width: int | None = None
    height: int | None = None


MediaAsset = AudioAsset | VideoAsset


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class TranscriptSegment(TimelineItem):
    """Segment-level transcript block."""

    text: str
    speaker: str | None = None


@dataclass(slots=True, frozen=True)
class TranscriptionResult:
    """Full normalized transcription output."""

    detected_language: str | None = None
    segments: list[TranscriptSegment] = dataclass_field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        """Return the transcript JSON artifact payload."""
        return {
            "detected_language": self.detected_language,
            "segments": [
                asdict(segment, dict_factory=_compact_dict_factory) for segment in self.segments
            ],
        }


# ---------------------------------------------------------------------------
# ASR planning and decode
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class SpeechRegion(TimelineSpan):
    """Speech-bearing region on the normalized audio timeline."""


@dataclass(slots=True, frozen=True)
class InferenceWindow(TimelineItem):
    """Window planned for one whisper.cpp inference call."""

    region_index: int


@dataclass(slots=True, frozen=True)
class DecodedWindow:
    """Transcript result for a planned inference window."""

    window: InferenceWindow
    input_prompt: str | None = None
    text: str = ""
    segments: list[TranscriptSegment] = dataclass_field(default_factory=list)
    detected_language: str | None = None

    def to_json(self) -> dict[str, object]:
        """Return the decoded-window JSON artifact payload."""
        return {
            "window": asdict(self.window),
            "input_prompt": self.input_prompt,
            "text": self.text,
            "segments": [
                asdict(segment, dict_factory=_compact_dict_factory) for segment in self.segments
            ],
            "detected_language": self.detected_language,
        }


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Scene(TimelineItem):
    """Time-bounded scene for video processing."""


@dataclass(slots=True, frozen=True)
class SceneFrame:
    """Representative frame selected from a scene."""

    id: str
    scene_id: str
    image_path: str
    timestamp_sec: float


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class SpeakerTurn(TimelineSpan):
    """Time-bounded speaker turn returned by diarization."""

    speaker: str


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ReportSection(TimelineItem):
    """Renderable section of the final report."""

    title: str
    transcript_text: str
    tldr: str | None = None
    scene_id: str | None = None
    frame_id: str | None = None
    image_path: str | None = None


@dataclass(slots=True, frozen=True)
class ReportDocument:
    """Top-level structured report returned by the pipeline."""

    title: str
    source_file: str
    media_type: MediaType
    detected_language: str | None = None
    summary: list[str] = dataclass_field(default_factory=list)
    action_items: list[str] = dataclass_field(default_factory=list)
    sections: list[ReportSection] = dataclass_field(default_factory=list)
    warnings: list[str] = dataclass_field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        """Return the report JSON artifact payload."""
        return asdict(self, dict_factory=_compact_dict_factory)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

ReportStatus = Literal["applied", "fallback"]


@dataclass(slots=True, frozen=True)
class AsrPipelineDiagnostics:
    """Collected ASR diagnostics state for one processing run."""

    backend: str | None
    model: str | None
    threads: int
    normalized_audio_duration_sec: float | None = None
    vad_region_count: int = 0
    window_count: int = 0
    average_window_duration_sec: float | None = None
    system_info: str | None = None


@dataclass(slots=True, frozen=True)
class DiarizationDiagnostics:
    """Collected speaker-diarization diagnostics for one processing run."""

    speaker_count: int
    turn_count: int
    model: str
    system_info: str | None = None


@dataclass(slots=True, frozen=True)
class LlmDiagnostics:
    """Collected LLM diagnostics for one LLM-enabled processing run."""

    model: str
    report_status: ReportStatus
    report_latency_sec: float
    response_metadata: list[dict[str, object]] = dataclass_field(default_factory=list)


@dataclass(slots=True, frozen=True)
class Diagnostics:
    """Execution metadata recorded for a processing run."""

    status: Literal["succeeded", "failed"] = "succeeded"
    failed_stage: str | None = None
    error: str | None = None
    llm: LlmDiagnostics | None = None
    stage_durations_sec: dict[str, float] = dataclass_field(default_factory=dict)
    item_counts: dict[str, int] = dataclass_field(default_factory=dict)
    asr_pipeline: AsrPipelineDiagnostics | None = None
    diarization: DiarizationDiagnostics | None = None
    warnings: list[str] = dataclass_field(default_factory=list)
