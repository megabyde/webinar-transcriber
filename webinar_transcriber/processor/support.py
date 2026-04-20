"""Shared helpers for processor orchestration and reporting."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

from webinar_transcriber.asr import ASR_BACKEND_NAME, WhisperCppTranscriber
from webinar_transcriber.labels import count_label, optional_count_label
from webinar_transcriber.models import (
    AsrPipelineDiagnostics,
    Diagnostics,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Literal

    from pydantic import BaseModel

    from webinar_transcriber.llm import LLMReportPolishPlan
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.reporter import NullStageReporter

    from .types import _AsrPipelineState


class StageContext(Protocol):
    """Shared stage context contract used across processor flows."""

    reporter: NullStageReporter
    stage_timings: dict[str, float]
    current_stage: str | None


@dataclass
class StageHandle:
    """Mutable stage state shared with the stage context manager body."""

    key: str
    label: str
    detail: str | None = None
    start_sec: float = 0.0

    def elapsed_sec(self) -> float:
        """Return the current elapsed stage time in seconds."""
        return perf_counter() - self.start_sec


@dataclass
class ProgressStageHandle(StageHandle):
    """Mutable progress-stage state shared with the stage context manager body."""

    reporter: NullStageReporter = field(kw_only=True)
    completed: float = 0.0

    def advance(self, advance: float = 1.0, *, detail: str | None = None) -> None:
        """Advance stage progress by one positive delta."""
        if advance <= 0:
            return
        self.reporter.progress_advanced(self.key, advance=advance, detail=detail)
        self.completed += advance
        if detail is not None:
            self.detail = detail

    def advance_to(self, completed: float, *, detail: str | None = None) -> None:
        """Advance stage progress up to one cumulative completed value."""
        self.advance(max(0.0, completed - self.completed), detail=detail)

    def finish_progress(self, total: float, *, detail: str | None = None) -> None:
        """Advance stage progress through one final total."""
        self.advance_to(total, detail=detail)


@contextmanager
def stage(
    ctx: StageContext, key: str, label: str, *, indeterminate: bool = True
) -> Iterator[StageHandle]:
    """Record one stage's timing and lifecycle through a context manager."""
    handle = StageHandle(key=key, label=label, start_sec=perf_counter())
    ctx.current_stage = key
    if indeterminate:
        ctx.reporter.stage_started(key, label)
    try:
        yield handle
    except Exception:
        ctx.stage_timings[key] = handle.elapsed_sec()
        raise
    ctx.stage_timings[key] = handle.elapsed_sec()
    ctx.reporter.stage_finished(key, label, detail=handle.detail)


@contextmanager
def progress_stage(
    ctx: StageContext,
    key: str,
    label: str,
    *,
    total: float,
    count_label: str | None = None,
    count_multiplier: float = 1.0,
    rate_label: str | None = None,
    rate_multiplier: float = 1.0,
    detail: str | None = None,
) -> Iterator[ProgressStageHandle]:
    """Record one determinate stage's timing and progress through a context manager."""
    handle = ProgressStageHandle(
        key=key,
        label=label,
        detail=detail,
        start_sec=perf_counter(),
        reporter=ctx.reporter,
    )
    ctx.current_stage = key
    ctx.reporter.progress_started(
        key,
        label,
        total=total,
        count_label=count_label,
        count_multiplier=count_multiplier,
        rate_label=rate_label,
        rate_multiplier=rate_multiplier,
        detail=detail,
    )
    try:
        yield handle
    except Exception:
        ctx.stage_timings[key] = handle.elapsed_sec()
        raise
    ctx.stage_timings[key] = handle.elapsed_sec()
    ctx.reporter.stage_finished(key, label, detail=handle.detail)


def write_json(output_path: Path, payload: dict[str, object]) -> None:
    """Write one JSON payload with stable UTF-8 formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_model_json(output_path: Path, payload: BaseModel) -> None:
    """Write one top-level Pydantic payload with stable UTF-8 formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def configure_asr_logging(transcriber: WhisperCppTranscriber, layout: RunLayout) -> None:
    """Write pywhispercpp logs into the run directory."""
    log_path = layout.run_dir / "whisper-cpp.log"
    logger = logging.getLogger("pywhispercpp")
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_path))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    transcriber.set_log_path(log_path)


def asr_runtime_detail(transcriber: WhisperCppTranscriber) -> str:
    """Return a human-facing ASR runtime label."""
    return f"{transcriber.model_name} | {transcriber.device_name}"


def window_transcription_stage_detail(
    *, window_count: int, total_duration_sec: float, elapsed_sec: float
) -> str:
    """Return the transcribe-stage summary with window count and real-time factor."""
    details = [count_label(window_count, "window")]
    if total_duration_sec > 0 and elapsed_sec > 0:
        realtime_multiple = f"{total_duration_sec / elapsed_sec:.2f}".rstrip("0").rstrip(".")
        details.append(f"RTF {realtime_multiple}x")
    return " | ".join(details)


def llm_runtime_detail(*, provider_name: str | None, model_name: str | None) -> str:
    """Return a compact provider/model label for the LLM runtime."""
    parts = [value for value in (provider_name, model_name) if value]
    return " | ".join(parts)


def llm_stage_label(
    base_label: str, *, provider_name: str | None, model_name: str | None, detail: str | None = None
) -> str:
    """Return one stage label decorated with provider/model details."""
    runtime_detail = llm_runtime_detail(provider_name=provider_name, model_name=model_name)
    parenthetical = " | ".join(part for part in (runtime_detail, detail) if part)
    return f"{base_label} ({parenthetical})" if parenthetical else base_label


def llm_report_detail(
    *,
    section_count: int,
    tldr_count: int,
    title_count: int,
    summary_count: int,
    action_item_count: int,
    usage: dict[str, int],
) -> str:
    """Return the summary detail string for the report-polish stage."""
    parts = [
        optional_count_label(summary_count, "summary bullet"),
        optional_count_label(action_item_count, "action item"),
        optional_count_label(tldr_count, "TL;DR"),
        title_update_detail(title_count=title_count, section_count=section_count),
        token_usage_detail(usage),
    ]
    return " | ".join(part for part in parts if part)


def llm_fallback_detail(*, provider_name: str | None, model_name: str | None) -> str:
    """Return the fallback detail string for failed LLM stages."""
    runtime_detail = llm_runtime_detail(provider_name=provider_name, model_name=model_name)
    return " | ".join(part for part in (runtime_detail, "fallback") if part)


def llm_report_plan_label_detail(plan: LLMReportPolishPlan) -> str:
    """Return the worker-count detail for the section-polish stage."""
    return count_label(plan.worker_count, "worker")


def token_usage_detail(usage: dict[str, int]) -> str:
    """Return a compact total-token label when token accounting is available."""
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        return ""
    return count_label(total_tokens, "token")


def title_update_detail(*, title_count: int, section_count: int) -> str:
    """Return a title-update detail only when some but not all titles changed."""
    if title_count <= 0 or title_count == section_count:
        return ""
    return count_label(title_count, "title updated", plural="titles updated")


def build_diagnostics(
    *,
    status: Literal["succeeded", "failed"] = "succeeded",
    failed_stage: str | None = None,
    error: str | None = None,
    asr_model: str | None,
    llm_enabled: bool,
    llm_model: str | None,
    llm_report_status: str,
    llm_report_latency_sec: float | None,
    llm_report_usage: dict[str, int] | None,
    stage_timings: dict[str, float],
    asr_pipeline: _AsrPipelineState,
    transcript_segment_count: int,
    normalized_transcript_segment_count: int,
    report_section_count: int,
    scene_count: int,
    frame_count: int,
    warnings: list[str],
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
        asr_model=asr_model,
        llm_enabled=llm_enabled,
        llm_model=llm_model,
        llm_report_status=llm_report_status,
        llm_report_latency_sec=llm_report_latency_sec,
        llm_report_usage=llm_report_usage or {},
        stage_durations_sec={key: round(value, 6) for key, value in stage_timings.items()},
        item_counts={
            "transcript_segments": transcript_segment_count,
            "normalized_transcript_segments": normalized_transcript_segment_count,
            "vad_regions": asr_pipeline.vad_region_count,
            "windows": asr_pipeline.window_count,
            "report_sections": report_section_count,
            "scenes": scene_count,
            "frames": frame_count,
        },
        asr_pipeline=AsrPipelineDiagnostics(**asdict(asr_pipeline)),
        warnings=warnings,
    )
