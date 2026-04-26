"""Run-diagnostics assembly and persistence helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

from webinar_transcriber.asr import ASR_BACKEND_NAME
from webinar_transcriber.models import AsrPipelineDiagnostics, Diagnostics

if TYPE_CHECKING:
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.processor.types import AsrPipelineState, RunContext


@dataclass(frozen=True)
class DiagnosticsState:
    """Collected run state needed to assemble diagnostics.json."""

    asr_model: str | None
    llm_enabled: bool
    llm_model: str | None
    llm_report_status: str
    llm_report_latency_sec: float | None
    llm_report_usage: dict[str, int] | None
    stage_timings: dict[str, float]
    asr_pipeline: AsrPipelineState
    transcript_segment_count: int
    normalized_transcript_segment_count: int
    report_section_count: int
    scene_count: int
    frame_count: int
    warnings: list[str]


def build_diagnostics(
    state: DiagnosticsState,
    *,
    status: Literal["succeeded", "failed"] = "succeeded",
    failed_stage: str | None = None,
    error: str | None = None,
) -> Diagnostics:
    """Build the final diagnostics payload for one processing run.

    Returns:
        Diagnostics: The final diagnostics payload.
    """
    return Diagnostics(
        status=status,
        failed_stage=failed_stage,
        error=error,
        asr_backend=ASR_BACKEND_NAME,
        asr_model=state.asr_model,
        llm_enabled=state.llm_enabled,
        llm_model=state.llm_model,
        llm_report_status=state.llm_report_status,
        llm_report_latency_sec=state.llm_report_latency_sec,
        llm_report_usage=state.llm_report_usage or {},
        stage_durations_sec={key: round(value, 6) for key, value in state.stage_timings.items()},
        item_counts={
            "transcript_segments": state.transcript_segment_count,
            "normalized_transcript_segments": state.normalized_transcript_segment_count,
            "vad_regions": state.asr_pipeline.vad_region_count,
            "windows": state.asr_pipeline.window_count,
            "report_sections": state.report_section_count,
            "scenes": state.scene_count,
            "frames": state.frame_count,
        },
        asr_pipeline=AsrPipelineDiagnostics(**asdict(state.asr_pipeline)),
        warnings=state.warnings,
    )


def write_diagnostics(
    layout: RunLayout | None,
    state: DiagnosticsState,
    *,
    status: Literal["succeeded", "failed"],
    failed_stage: str | None = None,
    error: str | None = None,
    suppress_errors: bool = False,
) -> Diagnostics | None:
    """Write diagnostics when a run layout exists and return the payload.

    Returns:
        Diagnostics | None: The written diagnostics payload, if a run layout exists.
    """
    if layout is None:
        return None

    diagnostics = build_diagnostics(
        state,
        status=status,
        failed_stage=failed_stage,
        error=error,
    )
    try:
        layout.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        layout.diagnostics_path.write_text(
            json.dumps(asdict(diagnostics), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        if not suppress_errors:
            raise
    return diagnostics


def write_run_diagnostics(
    ctx: RunContext,
    *,
    status: Literal["succeeded", "failed"],
    asr_model: str | None,
    llm_enabled: bool,
    failed_stage: str | None = None,
    error: str | None = None,
    suppress_errors: bool = False,
) -> Diagnostics | None:
    """Write diagnostics for the current processor context."""
    return write_diagnostics(
        ctx.layout,
        DiagnosticsState(
            asr_model=asr_model,
            llm_enabled=llm_enabled,
            llm_model=ctx.llm_runtime.model_name,
            llm_report_status=ctx.llm_runtime.report_status,
            llm_report_latency_sec=ctx.llm_runtime.report_latency_sec,
            llm_report_usage=ctx.llm_runtime.report_usage,
            stage_timings=ctx.stage_timings,
            asr_pipeline=ctx.asr_pipeline,
            transcript_segment_count=len(ctx.transcription.segments) if ctx.transcription else 0,
            normalized_transcript_segment_count=(
                len(ctx.normalized_transcription.segments) if ctx.normalized_transcription else 0
            ),
            report_section_count=len(ctx.report.sections) if ctx.report else 0,
            scene_count=len(ctx.scenes),
            frame_count=len(ctx.slide_frames),
            warnings=ctx.warnings,
        ),
        status=status,
        failed_stage=failed_stage,
        error=error,
        suppress_errors=suppress_errors,
    )
