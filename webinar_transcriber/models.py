"""Typed models used across the processing pipeline."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class MediaType(StrEnum):
    """Supported top-level media types."""

    AUDIO = "audio"
    VIDEO = "video"


class _BaseMediaAsset(BaseModel):
    """Shared metadata for a probed media asset."""

    path: str
    duration_sec: float = Field(ge=0)
    sample_rate: int | None = Field(default=None, ge=1)
    channels: int | None = Field(default=None, ge=1)


class AudioAsset(_BaseMediaAsset):
    """Metadata for a probed audio-only asset."""

    media_type: Literal[MediaType.AUDIO] = MediaType.AUDIO


class VideoAsset(_BaseMediaAsset):
    """Metadata for a probed video asset."""

    media_type: Literal[MediaType.VIDEO] = MediaType.VIDEO
    fps: float | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)


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


class TranscriptionResult(BaseModel):
    """Full normalized transcription output."""

    model_config = {"arbitrary_types_allowed": True}

    detected_language: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)


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
    fallback_used: bool = False
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


class ReportSection(BaseModel):
    """Renderable section of the final report."""

    id: str
    title: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    tldr: str | None = None
    transcript_text: str
    frame_id: str | None = None
    image_path: str | None = None


class ReportDocument(BaseModel):
    """Top-level structured report returned by the pipeline."""

    title: str
    source_file: str
    media_type: MediaType
    detected_language: str | None = None
    summary: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AsrPipelineDiagnostics(BaseModel):
    """Additional telemetry for the windowed ASR pipeline."""

    normalized_audio_duration_sec: float | None = Field(default=None, ge=0)
    vad_enabled: bool = False
    vad_region_count: int = Field(default=0, ge=0)
    carryover_enabled: bool = False
    window_count: int = Field(default=0, ge=0)
    average_window_duration_sec: float | None = Field(default=None, ge=0)
    reconciliation_duplicate_segments_dropped: int = Field(default=0, ge=0)
    reconciliation_boundary_fixes: int = Field(default=0, ge=0)
    threads: int | None = Field(default=None, ge=1)
    system_info: str | None = None


class Diagnostics(BaseModel):
    """Execution metadata recorded for a processing run."""

    status: Literal["succeeded", "failed"] = "succeeded"
    failed_stage: str | None = None
    error: str | None = None
    asr_backend: str | None = None
    asr_model: str | None = None
    llm_enabled: bool = False
    llm_model: str | None = None
    llm_report_status: str = "disabled"
    llm_report_latency_sec: float | None = None
    llm_report_usage: dict[str, int] = Field(default_factory=dict)
    stage_durations_sec: dict[str, float] = Field(default_factory=dict)
    item_counts: dict[str, int] = Field(default_factory=dict)
    asr_pipeline: AsrPipelineDiagnostics | None = None
    warnings: list[str] = Field(default_factory=list)
