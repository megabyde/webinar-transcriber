"""Shared helpers for processor orchestration and reporting."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from webinar_transcriber.asr import WhisperCppTranscriber
    from webinar_transcriber.reporter import BaseStageReporter

    from .types import RunContext


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

    reporter: BaseStageReporter = field(kw_only=True)
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


def count_label(count: int, singular: str, *, plural: str | None = None) -> str:
    """Return one compact count label with naive English pluralization."""
    noun = singular if count == 1 else plural or f"{singular}s"
    return f"{count} {noun}"


def _count_label_if_positive(count: int, singular: str, *, plural: str | None = None) -> str | None:
    return count_label(count, singular, plural=plural) if count > 0 else None


def _detail_label(*parts: str | None) -> str:
    return " | ".join(part for part in parts if part)


@contextmanager
def stage(
    ctx: RunContext, key: str, label: str, *, indeterminate: bool = True
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
    ctx: RunContext,
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


def asr_runtime_detail(transcriber: WhisperCppTranscriber) -> str:
    """Return a human-facing ASR runtime label."""
    return f"{transcriber.model_name} | {transcriber.device_name}"


def window_transcription_stage_detail(
    *, window_count: int, total_duration_sec: float, elapsed_sec: float
) -> str:
    """Return the transcribe-stage summary with window count and real-time factor."""
    details = [count_label(window_count, "window")]
    if total_duration_sec > 0 and elapsed_sec > 0:
        realtime_multiple = format(round(total_duration_sec / elapsed_sec, 2), "g")
        details.append(f"RTF {realtime_multiple}x")
    return " | ".join(details)


def _llm_runtime_detail(*, provider_name: str | None, model_name: str | None) -> str:
    return _detail_label(provider_name, model_name)


def llm_stage_label(
    base_label: str, *, provider_name: str | None, model_name: str | None, detail: str | None = None
) -> str:
    """Return one stage label decorated with provider/model details."""
    runtime_detail = _llm_runtime_detail(provider_name=provider_name, model_name=model_name)
    parenthetical = _detail_label(runtime_detail, detail)
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
    title_update_count = title_count if title_count > 0 and title_count != section_count else 0
    return _detail_label(
        _count_label_if_positive(summary_count, "summary bullet"),
        _count_label_if_positive(action_item_count, "action item"),
        _count_label_if_positive(tldr_count, "TL;DR"),
        _count_label_if_positive(
            title_update_count,
            "title updated",
            plural="titles updated",
        ),
        _count_label_if_positive(usage.get("total_tokens", 0), "token"),
    )


def llm_fallback_detail(*, provider_name: str | None, model_name: str | None) -> str:
    """Return the fallback detail string for failed LLM stages."""
    runtime_detail = _llm_runtime_detail(provider_name=provider_name, model_name=model_name)
    return _detail_label(runtime_detail, "fallback")
