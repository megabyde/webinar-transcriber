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


class TranscriptSegment(BaseModel):
    """Segment-level transcript block."""

    id: str
    text: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)


class TranscriptionResult(BaseModel):
    """Full normalized transcription output."""

    detected_language: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)


class SpeechRegion(BaseModel):
    """Speech-bearing region on the normalized audio timeline."""

    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)


class AudioChunk(BaseModel):
    """Chunk planned for ASR inference."""

    id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)


class ChunkTranscription(BaseModel):
    """Transcript result for a planned audio chunk."""

    chunk_id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
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


class AlignmentBlock(BaseModel):
    """Alignment between transcript content and media sections."""

    id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    transcript_segment_ids: list[str] = Field(default_factory=list)
    transcript_text: str
    scene_id: str | None = None
    frame_id: str | None = None
    title_hint: str | None = None
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
    """Additional telemetry for the chunked ASR pipeline."""

    normalized_audio_duration_sec: float | None = Field(default=None, ge=0)
    vad_enabled: bool = False
    vad_region_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    average_chunk_duration_sec: float | None = Field(default=None, ge=0)
    overlap_duration_sec: float | None = Field(default=None, ge=0)
    reconciliation_duplicate_segments_dropped: int = Field(default=0, ge=0)
    reconciliation_boundary_fixes: int = Field(default=0, ge=0)
    threads: int | None = Field(default=None, ge=1)
    system_info: str | None = None


class Diagnostics(BaseModel):
    """Execution metadata recorded for a processing run."""

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
