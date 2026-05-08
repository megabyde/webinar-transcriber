"""Run-diagnostics assembly and persistence helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Literal

from webinar_transcriber.asr import ASR_BACKEND_NAME
from webinar_transcriber.models import AsrPipelineDiagnostics, Diagnostics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from webinar_transcriber.models import (
        ReportDocument,
        Scene,
        SlideFrame,
        TranscriptionResult,
    )
    from webinar_transcriber.processor.types import RunContext


def build_diagnostics(
    ctx: RunContext,
    *,
    asr_model: str | None,
    llm_enabled: bool,
    transcription: TranscriptionResult | None = None,
    normalized_transcription: TranscriptionResult | None = None,
    asr_pipeline: AsrPipelineDiagnostics | None = None,
    report: ReportDocument | None = None,
    scenes: Sequence[Scene] = (),
    slide_frames: Sequence[SlideFrame] = (),
    status: Literal["succeeded", "failed"] = "succeeded",
    failed_stage: str | None = None,
    error: str | None = None,
) -> Diagnostics:
    """Build the final diagnostics payload for one processing run.

    Returns:
        Diagnostics: The final diagnostics payload.
    """
    vad_region_count = asr_pipeline.vad_region_count if asr_pipeline is not None else 0
    window_count = asr_pipeline.window_count if asr_pipeline is not None else 0
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
            "transcript_segments": len(transcription.segments) if transcription else 0,
            "normalized_transcript_segments": (
                len(normalized_transcription.segments) if normalized_transcription else 0
            ),
            "vad_regions": vad_region_count,
            "windows": window_count,
            "report_sections": len(report.sections) if report else 0,
            "scenes": len(scenes),
            "frames": len(slide_frames),
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
    transcription: TranscriptionResult | None = None,
    normalized_transcription: TranscriptionResult | None = None,
    asr_pipeline: AsrPipelineDiagnostics | None = None,
    report: ReportDocument | None = None,
    scenes: Sequence[Scene] = (),
    slide_frames: Sequence[SlideFrame] = (),
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
        transcription=transcription,
        normalized_transcription=normalized_transcription,
        asr_pipeline=asr_pipeline,
        report=report,
        scenes=scenes,
        slide_frames=slide_frames,
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
