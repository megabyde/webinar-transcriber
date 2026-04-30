"""Shared processor dataclasses and phase results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .llm_types import LLMRuntimeState

if TYPE_CHECKING:
    from webinar_transcriber.models import (
        AlignmentBlock,
        AsrPipelineDiagnostics,
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


@dataclass
class RunContext:
    """Mutable state for one processing run."""

    reporter: BaseStageReporter
    stage_timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    current_stage: str | None = None
    layout: RunLayout | None = None
    llm_runtime: LLMRuntimeState = field(default_factory=LLMRuntimeState)


@dataclass(frozen=True)
class TranscriptionPhaseResult:
    """Artifacts produced by the transcription half of the pipeline."""

    transcription: TranscriptionResult
    normalized_transcription: TranscriptionResult
    asr_pipeline: AsrPipelineDiagnostics


@dataclass(frozen=True)
class ReportPhaseResult:
    """Artifacts produced by the report half of the pipeline."""

    report: ReportDocument
    alignment_blocks: list[AlignmentBlock] | None
    scenes: list[Scene]
    slide_frames: list[SlideFrame]
