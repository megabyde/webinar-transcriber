"""LLM runtime helpers used by the top-level processor orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from webinar_transcriber.labels import count_label
from webinar_transcriber.llm import (
    LLMConfigurationError,
    LLMProcessingError,
    build_llm_processor_from_env,
)
from webinar_transcriber.usage import merge_usage

from .support import (
    llm_fallback_detail,
    llm_report_detail,
    llm_report_plan_label_detail,
    llm_stage_label,
    start_stage_timer,
)

if TYPE_CHECKING:
    from webinar_transcriber.llm import LLMProcessor
    from webinar_transcriber.models import ReportDocument
    from webinar_transcriber.reporter import StageReporter


@dataclass
class LLMRuntimeState:
    """Observed state for the optional LLM report stage."""

    provider_name: str | None = None
    model_name: str | None = None
    report_status: str = "disabled"
    report_latency_sec: float | None = None
    report_usage: dict[str, int] | None = None


def resolve_llm_processor(
    *,
    enable_llm: bool,
    llm_processor: LLMProcessor | None,
    reporter: StageReporter,
    warnings: list[str],
    llm_runtime: LLMRuntimeState,
) -> tuple[LLMProcessor | None, LLMRuntimeState]:
    """Resolve the optional LLM processor and record configuration failures as warnings."""
    if not enable_llm:
        return None, llm_runtime

    if llm_processor is not None:
        llm_runtime.provider_name = llm_processor.provider_name
        llm_runtime.model_name = llm_processor.model_name
        return llm_processor, llm_runtime

    try:
        resolved_processor = build_llm_processor_from_env()
    except LLMConfigurationError as error:
        warnings.append(str(error))
        reporter.warn(str(error))
        llm_runtime.report_status = "fallback"
        return None, llm_runtime

    llm_runtime.provider_name = resolved_processor.provider_name
    llm_runtime.model_name = resolved_processor.model_name
    return resolved_processor, llm_runtime


def maybe_polish_report(
    report: ReportDocument,
    *,
    llm_processor: LLMProcessor | None,
    reporter: StageReporter,
    warnings: list[str],
    stage_timings: dict[str, float],
    llm_runtime: LLMRuntimeState,
) -> tuple[ReportDocument, LLMRuntimeState]:
    """Optionally run the LLM report-polish flow and record warnings/fallbacks."""
    if llm_processor is None:
        return report, llm_runtime

    polish_plan = llm_processor.report_polish_plan(report)
    section_label = llm_stage_label(
        "Polishing section text with LLM",
        provider_name=llm_runtime.provider_name,
        model_name=llm_runtime.model_name,
        detail=llm_report_plan_label_detail(polish_plan),
    )
    summary_label = llm_stage_label(
        "Polishing report summary with LLM",
        provider_name=llm_runtime.provider_name,
        model_name=llm_runtime.model_name,
    )
    reporter.progress_started(
        "llm_report_sections",
        section_label,
        total=max(float(polish_plan.section_count), 1.0),
        count_label="sections",
    )
    timer = start_stage_timer(stage_timings, "llm_report_sections")
    try:
        section_result = llm_processor.polish_report_sections_with_progress(
            report,
            progress_callback=lambda advance: reporter.progress_advanced(
                "llm_report_sections",
                advance=float(advance),
            ),
        )
    except LLMProcessingError as error:
        report_latency_sec = timer.finish()
        warnings.append(str(error))
        reporter.warn(str(error))
        reporter.stage_finished(
            "llm_report_sections",
            section_label,
            detail=llm_fallback_detail(
                provider_name=llm_runtime.provider_name,
                model_name=llm_runtime.model_name,
            ),
        )
        llm_runtime.report_status = "fallback"
        llm_runtime.report_latency_sec = report_latency_sec
        return report, llm_runtime

    section_elapsed_sec = timer.finish()
    for warning in section_result.warnings:
        warnings.append(warning)
        reporter.warn(warning)
    section_detail = count_label(polish_plan.section_count, "section")
    if polish_plan.skipped_section_count > 0:
        section_detail = " | ".join((
            section_detail,
            count_label(polish_plan.skipped_section_count, "skipped interlude"),
        ))
    reporter.stage_finished(
        "llm_report_sections",
        section_label,
        detail=section_detail,
    )
    reporter.stage_started("llm_report", summary_label)
    timer = start_stage_timer(stage_timings, "llm_report_metadata")

    try:
        metadata_result = llm_processor.polish_report_metadata(
            report,
            section_transcripts=section_result.section_transcripts,
        )
    except LLMProcessingError as error:
        metadata_elapsed_sec = timer.finish()
        report_latency_sec = section_elapsed_sec + metadata_elapsed_sec
        warnings.append(str(error))
        reporter.warn(str(error))
        reporter.stage_finished(
            "llm_report",
            summary_label,
            detail=llm_fallback_detail(
                provider_name=llm_runtime.provider_name,
                model_name=llm_runtime.model_name,
            ),
        )
        llm_runtime.report_status = "fallback"
        llm_runtime.report_latency_sec = report_latency_sec
        return report, llm_runtime

    metadata_elapsed_sec = timer.finish()
    report_latency_sec = section_elapsed_sec + metadata_elapsed_sec
    usage = merge_usage(section_result.usage, metadata_result.usage)
    report.summary = metadata_result.summary
    report.action_items = metadata_result.action_items
    for section in report.sections:
        section.title = metadata_result.section_titles.get(section.id, section.title)
        section.tldr = section_result.section_tldrs.get(section.id, section.tldr)
        section.transcript_text = section_result.section_transcripts.get(
            section.id,
            section.transcript_text,
        )
    reporter.stage_finished(
        "llm_report",
        summary_label,
        detail=llm_report_detail(
            section_count=polish_plan.section_count,
            tldr_count=len(section_result.section_tldrs),
            title_count=len(metadata_result.section_titles),
            summary_count=len(metadata_result.summary),
            action_item_count=len(metadata_result.action_items),
            usage=usage,
        ),
    )
    llm_runtime.report_status = "applied"
    llm_runtime.report_latency_sec = report_latency_sec
    llm_runtime.report_usage = usage
    return report, llm_runtime
