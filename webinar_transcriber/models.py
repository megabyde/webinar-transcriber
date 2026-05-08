"""Typed models used across the processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from typing import Literal


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


@dataclass(slots=True, frozen=True)
class TranscriptSegment:
    """Segment-level transcript block."""

    id: str
    text: str
    start_sec: float
    end_sec: float

    @property
    def midpoint(self) -> float:
        """Return the midpoint of the segment on the transcript timeline."""
        return (self.start_sec + self.end_sec) / 2.0


@dataclass(slots=True, frozen=True)
class TranscriptionResult:
    """Full normalized transcription output."""

    detected_language: str | None = None
    segments: list[TranscriptSegment] = dataclass_field(default_factory=list)


@dataclass(slots=True, frozen=True)
class SpeechRegion:
    """Speech-bearing region on the normalized audio timeline."""

    start_sec: float
    end_sec: float


@dataclass(slots=True, frozen=True)
class InferenceWindow:
    """Window planned for one whisper.cpp inference call."""

    window_id: str
    region_index: int
    start_sec: float
    end_sec: float


@dataclass(slots=True, frozen=True)
class DecodedWindow:
    """Transcript result for a planned inference window."""

    window: InferenceWindow
    input_prompt: str | None = None
    text: str = ""
    segments: list[TranscriptSegment] = dataclass_field(default_factory=list)
    language: str | None = None


@dataclass(slots=True, frozen=True)
class Scene:
    """Time-bounded scene for video processing."""

    id: str
    start_sec: float
    end_sec: float

    @property
    def midpoint(self) -> float:
        """Return the midpoint of the scene on the media timeline."""
        return (self.start_sec + self.end_sec) / 2.0


@dataclass(slots=True, frozen=True)
class SlideFrame:
    """Representative frame selected from a scene."""

    id: str
    scene_id: str
    image_path: str
    timestamp_sec: float


@dataclass(slots=True, frozen=True)
class AlignmentBlock:
    """Alignment between transcript content and media sections."""

    id: str
    start_sec: float
    end_sec: float
    transcript_text: str
    transcript_segment_ids: list[str] = dataclass_field(default_factory=list)
    scene_id: str | None = None
    frame_id: str | None = None


@dataclass(slots=True, frozen=True)
class ReportSection:
    """Renderable section of the final report."""

    id: str
    title: str
    start_sec: float
    end_sec: float
    transcript_text: str
    tldr: str | None = None
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


@dataclass(slots=True, frozen=True)
class AsrPipelineDiagnostics:
    """Collected ASR diagnostics state for one processing run."""

    vad_enabled: bool
    threads: int
    normalized_audio_duration_sec: float | None = None
    vad_region_count: int = 0
    carryover_enabled: bool = False
    window_count: int = 0
    average_window_duration_sec: float | None = None
    system_info: str | None = None


@dataclass(slots=True, frozen=True)
class Diagnostics:
    """Execution metadata recorded for a processing run."""

    status: Literal["succeeded", "failed"] = "succeeded"
    failed_stage: str | None = None
    error: str | None = None
    asr_backend: str | None = None
    asr_model: str | None = None
    llm_enabled: bool = False
    llm_model: str | None = None
    llm_report_status: Literal["disabled", "applied", "fallback"] = "disabled"
    llm_report_latency_sec: float | None = None
    llm_report_usage: dict[str, int] = dataclass_field(default_factory=dict)
    stage_durations_sec: dict[str, float] = dataclass_field(default_factory=dict)
    item_counts: dict[str, int] = dataclass_field(default_factory=dict)
    asr_pipeline: AsrPipelineDiagnostics | None = None
    warnings: list[str] = dataclass_field(default_factory=list)
