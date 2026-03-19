"""Typed models used across the processing pipeline."""

from enum import StrEnum

from pydantic import BaseModel, Field


class MediaType(StrEnum):
    """Supported top-level media types."""

    AUDIO = "audio"
    VIDEO = "video"


class MediaAsset(BaseModel):
    """Metadata for a probed audio or video asset."""

    path: str
    media_type: MediaType
    duration_sec: float = Field(ge=0)
    sample_rate: int | None = Field(default=None, ge=1)
    channels: int | None = Field(default=None, ge=1)
    fps: float | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)


class TranscriptWord(BaseModel):
    """Word-level transcription timing."""

    text: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)


class TranscriptSegment(BaseModel):
    """Segment-level transcript block."""

    id: str
    text: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    words: list[TranscriptWord] = Field(default_factory=list)


class TranscriptionResult(BaseModel):
    """Full normalized transcription output."""

    detected_language: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)


class Scene(BaseModel):
    """Time-bounded scene for video processing."""

    id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)


class SlideFrame(BaseModel):
    """Representative frame selected from a scene."""

    id: str
    scene_id: str
    image_path: str
    timestamp_sec: float = Field(ge=0)
    sharpness_score: float | None = None
    dedupe_hash: str | None = None


class OcrResult(BaseModel):
    """OCR output associated with a frame."""

    frame_id: str
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    language: str | None = None


class AlignmentBlock(BaseModel):
    """Alignment between transcript content and media sections."""

    id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    transcript_segment_ids: list[str] = Field(default_factory=list)
    transcript_text: str
    scene_id: str | None = None
    frame_id: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    """Renderable section of the final report."""

    id: str
    title: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    transcript_text: str
    bullet_points: list[str] = Field(default_factory=list)
    frame_id: str | None = None


class ReportDocument(BaseModel):
    """Top-level structured report returned by the pipeline."""

    title: str
    source_file: str
    media_type: MediaType
    detected_language: str | None = None
    ocr_enabled: bool = False
    summary: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Diagnostics(BaseModel):
    """Execution metadata recorded for a processing run."""

    stage_durations_sec: dict[str, float] = Field(default_factory=dict)
    item_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
