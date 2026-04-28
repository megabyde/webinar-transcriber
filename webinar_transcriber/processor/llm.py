"""LLM runtime helpers used by the top-level processor orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from webinar_transcriber.llm import (
    LLMConfigurationError,
    LLMProcessingError,
    build_llm_processor_from_env,
)

from .support import (
    count_label,
    llm_fallback_detail,
    llm_report_detail,
    llm_stage_label,
    progress_stage,
    stage,
)

if TYPE_CHECKING:
    from webinar_transcriber.llm import LLMProcessor
    from webinar_transcriber.models import ReportDocument
    from webinar_transcriber.reporter import BaseStageReporter

    from .support import ProgressStageHandle
    from .types import RunContext


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
    reporter: BaseStageReporter,
    warnings: list[str],
    llm_runtime: LLMRuntimeState,
) -> tuple[LLMProcessor | None, LLMRuntimeState]:
    """Resolve the optional LLM processor and record configuration failures as warnings.

    Returns:
        tuple[LLMProcessor | None, LLMRuntimeState]: The resolved processor and updated runtime.
    """
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
    ctx: RunContext,
    warnings: list[str],
    llm_runtime: LLMRuntimeState,
) -> tuple[ReportDocument, LLMRuntimeState]:
    """Optionally run the LLM report-polish flow and record warnings/fallbacks.

    Returns:
        tuple[ReportDocument, LLMRuntimeState]: The final report and updated LLM runtime state.
    """
    if llm_processor is None:
        return report, llm_runtime

    polish_plan = llm_processor.report_polish_plan(report)
    section_label = llm_stage_label(
        "Polishing section text with LLM",
        provider_name=llm_runtime.provider_name,
        model_name=llm_runtime.model_name,
        detail=count_label(polish_plan.worker_count, "worker"),
    )
    summary_label = llm_stage_label(
        "Polishing report summary with LLM",
        provider_name=llm_runtime.provider_name,
        model_name=llm_runtime.model_name,
    )
    with progress_stage(
        ctx,
        "llm_report_sections",
        section_label,
        total=max(float(polish_plan.section_count), 1.0),
        count_label="sections",
    ) as st:

        def on_section_progress(advance: int, handle: ProgressStageHandle = st) -> None:
            handle.advance(float(advance))

        try:
            section_result = llm_processor.polish_report_sections_with_progress(
                report,
                progress_callback=on_section_progress,
            )
        except LLMProcessingError as error:
            warnings.append(str(error))
            ctx.reporter.warn(str(error))
            st.detail = llm_fallback_detail(
                provider_name=llm_runtime.provider_name, model_name=llm_runtime.model_name
            )
            llm_runtime.report_status = "fallback"
            llm_runtime.report_latency_sec = st.elapsed_sec()
            return report, llm_runtime
        section_elapsed_sec = st.elapsed_sec()
        st.detail = count_label(polish_plan.section_count, "section")
    for warning in section_result.warnings:
        warnings.append(warning)
        ctx.reporter.warn(warning)

    metadata_error: LLMProcessingError | None = None
    try:
        with stage(ctx, "llm_report", summary_label) as st:
            try:
                metadata_result = llm_processor.polish_report_metadata(
                    report, section_transcripts=section_result.section_transcripts
                )
            except LLMProcessingError as error:
                warnings.append(str(error))
                ctx.reporter.warn(str(error))
                st.detail = llm_fallback_detail(
                    provider_name=llm_runtime.provider_name, model_name=llm_runtime.model_name
                )
                metadata_elapsed_sec = st.elapsed_sec()
                metadata_error = error
            else:
                metadata_elapsed_sec = st.elapsed_sec()
                usage = dict(Counter(section_result.usage) + Counter(metadata_result.usage))
                st.detail = llm_report_detail(
                    section_count=polish_plan.section_count,
                    tldr_count=len(section_result.section_tldrs),
                    title_count=len(metadata_result.section_titles),
                    summary_count=len(metadata_result.summary),
                    action_item_count=len(metadata_result.action_items),
                    usage=usage,
                )
    finally:
        if "llm_report" in ctx.stage_timings:
            ctx.stage_timings["llm_report_metadata"] = ctx.stage_timings.pop("llm_report")

    if metadata_error is not None:
        llm_runtime.report_status = "fallback"
        llm_runtime.report_latency_sec = section_elapsed_sec + metadata_elapsed_sec
        return report, llm_runtime

    report_latency_sec = section_elapsed_sec + metadata_elapsed_sec
    report = replace(
        report,
        summary=metadata_result.summary,
        action_items=metadata_result.action_items,
        sections=[
            replace(
                section,
                title=metadata_result.section_titles.get(section.id, section.title),
                tldr=section_result.section_tldrs.get(section.id, section.tldr),
                transcript_text=section_result.section_transcripts.get(
                    section.id, section.transcript_text
                ),
            )
            for section in report.sections
        ],
    )
    llm_runtime.report_status = "applied"
    llm_runtime.report_latency_sec = report_latency_sec
    llm_runtime.report_usage = usage
    return report, llm_runtime
