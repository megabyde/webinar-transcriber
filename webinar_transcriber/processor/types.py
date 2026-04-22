"""Shared processor dataclasses and phase results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .llm import LLMRuntimeState

if TYPE_CHECKING:
    from webinar_transcriber.models import (
        AlignmentBlock,
        Diagnostics,
        MediaAsset,
        ReportDocument,
        Scene,
        SlideFrame,
        TranscriptionResult,
    )
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.reporter import BaseStageReporter


@dataclass(frozen=True)
class ProcessArtifacts:
    """Runtime artifacts returned from a processing run."""

    layout: RunLayout
    media_asset: MediaAsset
    transcription: TranscriptionResult
    report: ReportDocument
    diagnostics: Diagnostics


@dataclass(frozen=True)
class AsrPipelineState:
    """Collected ASR diagnostics state for one processing run."""

    vad_enabled: bool
    threads: int
    normalized_audio_duration_sec: float | None = None
    vad_region_count: int = 0
    carryover_enabled: bool = False
    window_count: int = 0
    average_window_duration_sec: float | None = None
    reconciliation_boundary_fixes: int = 0
    system_info: str | None = None


@dataclass
class RunContext:
    """Mutable state for one processing run."""

    reporter: BaseStageReporter
    asr_pipeline: AsrPipelineState
    stage_timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    current_stage: str | None = None
    layout: RunLayout | None = None
    media_asset: MediaAsset | None = None
    alignment_blocks: list[AlignmentBlock] | None = None
    scenes: list[Scene] = field(default_factory=list)
    slide_frames: list[SlideFrame] = field(default_factory=list)
    transcription: TranscriptionResult | None = None
    normalized_transcription: TranscriptionResult | None = None
    report: ReportDocument | None = None
    llm_runtime: LLMRuntimeState = field(default_factory=LLMRuntimeState)


@dataclass(frozen=True)
class TranscriptionPhaseResult:
    """Artifacts produced by the transcription half of the pipeline."""

    transcription: TranscriptionResult
    normalized_transcription: TranscriptionResult
    asr_pipeline: AsrPipelineState


@dataclass(frozen=True)
class ReportPhaseResult:
    """Artifacts produced by the report half of the pipeline."""

    report: ReportDocument
    alignment_blocks: list[AlignmentBlock] | None
    scenes: list[Scene]
    slide_frames: list[SlideFrame]
