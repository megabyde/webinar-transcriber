"""Shared helpers for processor orchestration and reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from webinar_transcriber.asr import ASR_BACKEND_NAME, WhisperCppTranscriber
from webinar_transcriber.export import (
    write_docx_report,
    write_json_report,
    write_markdown_report,
    write_vtt_subtitles,
)
from webinar_transcriber.models import (
    AsrPipelineDiagnostics,
    Diagnostics,
    ReportDocument,
    TranscriptionResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from webinar_transcriber.llm import LLMReportPolishPlan
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.reporter import StageReporter


@dataclass(frozen=True)
class StageTimer:
    """Simple stage timer that records elapsed time into one shared mapping."""

    stage_timings: dict[str, float]
    key: str
    start_sec: float

    def finish(self) -> float:
        """Record elapsed time and return it."""
        elapsed_sec = perf_counter() - self.start_sec
        self.stage_timings[self.key] = elapsed_sec
        return elapsed_sec


def start_stage_timer(stage_timings: dict[str, float], key: str) -> StageTimer:
    """Start a stage timer backed by one shared timings map."""
    return StageTimer(stage_timings=stage_timings, key=key, start_sec=perf_counter())


def write_requested_artifacts(
    report: ReportDocument,
    transcription: TranscriptionResult,
    layout: RunLayout,
    output_format: str,
) -> None:
    """Write the requested human-facing artifacts plus canonical JSON/VTT outputs."""
    formats = {"md", "docx", "json"} if output_format == "all" else {output_format}

    if "md" in formats:
        write_markdown_report(report, layout.markdown_report_path)
    if "docx" in formats:
        write_docx_report(report, layout.docx_report_path)

    # Always write JSON - it is the canonical machine-readable artifact.
    write_json_report(report, layout.json_report_path)
    write_vtt_subtitles(transcription, layout.subtitle_vtt_path)


def write_json(output_path: Path, payload: dict[str, object]) -> None:
    """Write one JSON payload with stable UTF-8 formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def progress_updater(
    reporter: StageReporter,
    *,
    stage_key: str,
) -> tuple[Callable[..., None], Callable[..., None]]:
    """Return progress update helpers that only advance monotonically."""
    completed = 0.0

    def update(next_completed: float, *, detail: str | None = None) -> None:
        nonlocal completed
        advance = max(0.0, next_completed - completed)
        if advance > 0:
            reporter.progress_advanced(stage_key, advance=advance, detail=detail)
            completed = next_completed

    def finish(total: float, *, detail: str | None = None) -> None:
        update(total, detail=detail)

    return update, finish


def count_label(count: int, *, singular: str, plural: str | None = None) -> str:
    """Return a compact singular/plural count label."""
    resolved_plural = plural or f"{singular}s"
    label = singular if count == 1 else resolved_plural
    return f"{count} {label}"


def configure_asr_logging(transcriber: WhisperCppTranscriber, layout: RunLayout) -> None:
    """Write whisper.cpp logs into the run directory."""
    transcriber.set_log_path(layout.run_dir / "whisper-cpp.log")


def asr_model_label(model_name: str) -> str:
    """Return a compact display label for the configured ASR model."""
    path = Path(model_name)
    if path.is_absolute():
        if repo_label := hf_cache_repo_label(path):
            return f"{repo_label}/{path.name} (HF cache)"
        return path.name
    return model_name


def hf_cache_repo_label(path: Path) -> str | None:
    """Return the Hugging Face repo label embedded in a cache path, if present."""
    for part in path.parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return None


def asr_runtime_detail(transcriber: WhisperCppTranscriber) -> str:
    """Return a human-facing ASR runtime label."""
    model_label = asr_model_label(transcriber.model_name)
    return f"{model_label} | {transcriber.device_name}"


def window_transcription_stage_detail(
    *,
    window_count: int,
    total_duration_sec: float,
    elapsed_sec: float,
) -> str:
    """Return the transcribe-stage summary with window count and real-time factor."""
    window_label = "window" if window_count == 1 else "windows"
    details = [f"{window_count} {window_label}"]
    if total_duration_sec > 0 and elapsed_sec > 0:
        details.append(f"RTF {elapsed_sec / total_duration_sec:.2f}")
    return " | ".join(details)


def llm_runtime_detail(*, provider_name: str | None, model_name: str | None) -> str:
    """Return a compact provider/model label for the LLM runtime."""
    parts = [value for value in (provider_name, model_name) if value]
    return " | ".join(parts)


def llm_stage_label(
    base_label: str,
    *,
    provider_name: str | None,
    model_name: str | None,
    detail: str | None = None,
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
        optional_count_detail(summary_count, singular="summary bullet", plural="summary bullets"),
        optional_count_detail(
            action_item_count,
            singular="action item",
            plural="action items",
        ),
        optional_count_detail(tldr_count, singular="TL;DR", plural="TL;DRs"),
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
    parts = [count_label(plan.worker_count, singular="worker")]
    if plan.skipped_section_count > 0:
        parts.append(count_label(plan.skipped_section_count, singular="skipped interlude"))
    return " | ".join(parts)


def token_usage_detail(usage: dict[str, int]) -> str:
    """Return a compact total-token label when token accounting is available."""
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        return ""
    token_label = "token" if total_tokens == 1 else "tokens"
    return f"{total_tokens} {token_label}"


def optional_count_detail(count: int, *, singular: str, plural: str) -> str:
    """Return a count label only when the count is positive."""
    if count <= 0:
        return ""
    label = singular if count == 1 else plural
    return f"{count} {label}"


def title_update_detail(*, title_count: int, section_count: int) -> str:
    """Return a title-update detail only when some but not all titles changed."""
    if title_count <= 0 or title_count == section_count:
        return ""
    label = "title updated" if title_count == 1 else "titles updated"
    return f"{title_count} {label}"


def build_diagnostics(
    *,
    status: str = "succeeded",
    failed_stage: str | None = None,
    error: str | None = None,
    asr_model: str | None,
    llm_enabled: bool,
    llm_model: str | None,
    llm_report_status: str,
    llm_report_latency_sec: float | None,
    llm_report_usage: dict[str, int] | None,
    stage_timings: dict[str, float],
    asr_pipeline: AsrPipelineDiagnostics,
    transcript_segment_count: int,
    normalized_transcript_segment_count: int,
    report_section_count: int,
    scene_count: int,
    frame_count: int,
    warnings: list[str],
) -> Diagnostics:
    """Build the final diagnostics payload for one processing run."""
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
        asr_pipeline=asr_pipeline,
        warnings=warnings,
    )
