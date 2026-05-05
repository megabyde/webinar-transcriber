"""Run-diagnostics assembly and persistence helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Literal

from webinar_transcriber.asr import ASR_BACKEND_NAME
from webinar_transcriber.models import AsrPipelineDiagnostics, Diagnostics

if TYPE_CHECKING:
    from webinar_transcriber.processor.types import (
        ReportPhaseResult,
        RunContext,
        TranscriptionPhaseResult,
    )


def build_diagnostics(
    ctx: RunContext,
    *,
    asr_model: str | None,
    llm_enabled: bool,
    transcription_phase: TranscriptionPhaseResult | None = None,
    report_phase: ReportPhaseResult | None = None,
    status: Literal["succeeded", "failed"] = "succeeded",
    failed_stage: str | None = None,
    error: str | None = None,
) -> Diagnostics:
    """Build the final diagnostics payload for one processing run.

    Returns:
        Diagnostics: The final diagnostics payload.
    """
    asr_pipeline = (
        transcription_phase.asr_pipeline
        if transcription_phase is not None
        else AsrPipelineDiagnostics(vad_enabled=False, threads=0)
    )
    return Diagnostics(
        status=status,
        failed_stage=failed_stage,
        error=error,
        asr_backend=ASR_BACKEND_NAME,
        asr_model=asr_model,
        llm_enabled=llm_enabled,
        llm_model=ctx.llm_runtime.model_name,
        llm_report_status=ctx.llm_runtime.report_status,
        llm_report_latency_sec=ctx.llm_runtime.report_latency_sec,
        llm_report_usage=ctx.llm_runtime.report_usage or {},
        stage_durations_sec={key: round(value, 6) for key, value in ctx.stage_timings.items()},
        item_counts={
            "transcript_segments": (
                len(transcription_phase.transcription.segments) if transcription_phase else 0
            ),
            "normalized_transcript_segments": (
                len(transcription_phase.normalized_transcription.segments)
                if transcription_phase
                else 0
            ),
            "vad_regions": asr_pipeline.vad_region_count,
            "windows": asr_pipeline.window_count,
            "report_sections": len(report_phase.report.sections) if report_phase else 0,
            "scenes": len(report_phase.scenes) if report_phase else 0,
            "frames": len(report_phase.slide_frames) if report_phase else 0,
        },
        asr_pipeline=asr_pipeline,
        warnings=ctx.warnings,
    )


def write_run_diagnostics(
    ctx: RunContext,
    *,
    status: Literal["succeeded", "failed"],
    asr_model: str | None,
    llm_enabled: bool,
    transcription_phase: TranscriptionPhaseResult | None = None,
    report_phase: ReportPhaseResult | None = None,
    failed_stage: str | None = None,
    error: str | None = None,
    suppress_errors: bool = False,
) -> Diagnostics | None:
    """Write diagnostics for the current processor context."""
    if ctx.layout is None:
        return None

    diagnostics = build_diagnostics(
        ctx,
        asr_model=asr_model,
        llm_enabled=llm_enabled,
        transcription_phase=transcription_phase,
        report_phase=report_phase,
        status=status,
        failed_stage=failed_stage,
        error=error,
    )
    try:
        ctx.layout.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.layout.diagnostics_path.write_text(
            json.dumps(asdict(diagnostics), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        if not suppress_errors:
            raise
    return diagnostics
