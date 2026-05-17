"""Shared processor dataclasses and phase results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .llm_types import LLMRuntimeState

if TYPE_CHECKING:
    from webinar_transcriber.diarization import Diarizer
    from webinar_transcriber.llm.contracts import LLMProcessor
    from webinar_transcriber.models import (
        Diagnostics,
        MediaAsset,
        ReportDocument,
        TranscriptionResult,
    )
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.reporter import BaseStageReporter


@dataclass(frozen=True)
class TranscriptionConfig:
    """Audio preparation and ASR options for one processing run."""

    threads: int
    asr_model: str | None = None
    language: str | None = None
    keep_audio: bool = False


@dataclass(frozen=True)
class LLMConfig:
    """Optional cloud LLM enhancement options for one processing run."""

    processor: LLMProcessor | Literal["from_env"] | None = None

    @property
    def enabled(self) -> bool:
        """Return whether LLM processing should run."""
        return self.processor is not None


@dataclass(frozen=True)
class DiarizationConfig:
    """Optional local speaker-diarization options for one processing run."""

    enabled: bool = False
    speaker_count: int | None = None
    diarizer: Diarizer | None = None


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

    def record_warning(self, message: str) -> None:
        """Record and report one run warning."""
        self.warnings.append(message)
        self.reporter.warn(message)
