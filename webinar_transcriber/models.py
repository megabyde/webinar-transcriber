"""Typed models used across the processing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from typing import Literal

from webinar_transcriber.json_utils import compact_speaker_fields


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

ReportStatus = Literal["disabled", "applied", "fallback"]


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
    start_sec: float
    end_sec: float


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
            "segments": [compact_speaker_fields(asdict(segment)) for segment in self.segments],
        }


@dataclass(slots=True, frozen=True)
class SpeechRegion(TimelineSpan):
    """Speech-bearing region on the normalized audio timeline."""

    start_sec: float
    end_sec: float


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
            "segments": [compact_speaker_fields(asdict(segment)) for segment in self.segments],
            "detected_language": self.detected_language,
        }


@dataclass(slots=True, frozen=True)
class Scene(TimelineItem):
    """Time-bounded scene for video processing."""


@dataclass(slots=True, frozen=True)
class SlideFrame:
    """Representative frame selected from a scene."""

    id: str
    scene_id: str
    image_path: str
    timestamp_sec: float


@dataclass(slots=True, frozen=True)
class VideoAssetRef:
    """Reference to video scene/frame context for report sections."""

    scene_id: str
    frame_id: str | None = None


@dataclass(slots=True, frozen=True)
class AlignmentBlock(TimelineItem):
    """Alignment between transcript content and media sections."""

    transcript_text: str
    transcript_segment_ids: list[str] = dataclass_field(default_factory=list)
    video: VideoAssetRef | None = None

    @property
    def scene_id(self) -> str | None:
        """Return the aligned scene id when available."""
        return self.video.scene_id if self.video else None

    @property
    def frame_id(self) -> str | None:
        """Return the aligned frame id when available."""
        return self.video.frame_id if self.video else None


@dataclass(slots=True, frozen=True)
class ReportSection(TimelineItem):
    """Renderable section of the final report."""

    title: str
    transcript_text: str
    tldr: str | None = None
    video: VideoAssetRef | None = None
    image_path: str | None = None
    speakers: list[str] = dataclass_field(default_factory=list)

    @property
    def scene_id(self) -> str | None:
        """Return the aligned scene id when available."""
        return self.video.scene_id if self.video else None

    @property
    def frame_id(self) -> str | None:
        """Return the aligned frame id when available."""
        return self.video.frame_id if self.video else None


@dataclass(slots=True, frozen=True)
class SpeakerTurn(TimelineSpan):
    """Time-bounded speaker turn returned by diarization."""

    start_sec: float
    end_sec: float
    speaker: str


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


@dataclass(slots=True, frozen=True)
class AsrPipelineDiagnostics:
    """Collected ASR diagnostics state for one processing run."""

    backend: str | None
    model: str | None
    vad_enabled: bool
    threads: int
    normalized_audio_duration_sec: float | None = None
    vad_region_count: int = 0
    carryover_enabled: bool = False
    window_count: int = 0
    average_window_duration_sec: float | None = None
    system_info: str | None = None


@dataclass(slots=True, frozen=True)
class DiarizationDiagnostics:
    """Collected speaker-diarization diagnostics for one processing run."""

    speaker_count: int
    turn_count: int
    average_turn_duration_sec: float | None
    model: str
    system_info: str | None = None


@dataclass(slots=True, frozen=True)
class LlmDiagnostics:
    """Collected optional LLM diagnostics for one processing run."""

    enabled: bool = False
    model: str | None = None
    report_status: ReportStatus = "disabled"
    report_latency_sec: float | None = None
    response_metadata: list[dict[str, object]] = dataclass_field(default_factory=list)


@dataclass(slots=True, frozen=True)
class Diagnostics:
    """Execution metadata recorded for a processing run."""

    status: Literal["succeeded", "failed"] = "succeeded"
    failed_stage: str | None = None
    error: str | None = None
    llm: LlmDiagnostics = dataclass_field(default_factory=LlmDiagnostics)
    stage_durations_sec: dict[str, float] = dataclass_field(default_factory=dict)
    item_counts: dict[str, int] = dataclass_field(default_factory=dict)
    asr_pipeline: AsrPipelineDiagnostics | None = None
    diarization: DiarizationDiagnostics | None = None
    warnings: list[str] = dataclass_field(default_factory=list)
