"""LLM runtime helpers used by the top-level processor orchestration."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from webinar_transcriber.llm import build_llm_processor_from_env
from webinar_transcriber.llm.contracts import (
    LLMConfigurationError,
    LLMProcessingError,
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
    from webinar_transcriber.llm.contracts import LLMProcessor
    from webinar_transcriber.models import ReportDocument

    from .llm_types import LLMRuntimeState
    from .support import ProgressStageHandle, StageHandle
    from .types import RunContext


def resolve_llm_processor(
    *,
    enable_llm: bool,
    llm_processor: LLMProcessor | None,
    ctx: RunContext,
    llm_runtime: LLMRuntimeState,
    threads: int,
) -> LLMProcessor | None:
    """Resolve the optional LLM processor and record configuration failures as warnings.

    Returns:
        LLMProcessor | None: The resolved processor when LLM processing is available.
    """
    if not enable_llm:
        return None

    if llm_processor is not None:
        llm_runtime.provider_name = llm_processor.provider_name
        llm_runtime.model_name = llm_processor.model_name
        return llm_processor

    try:
        resolved_processor = build_llm_processor_from_env(threads=threads)
    except LLMConfigurationError as error:
        ctx.record_warning(str(error))
        llm_runtime.report_status = "fallback"
        return None

    llm_runtime.provider_name = resolved_processor.provider_name
    llm_runtime.model_name = resolved_processor.model_name
    return resolved_processor


def maybe_polish_report(
    report: ReportDocument,
    *,
    llm_processor: LLMProcessor | None,
    ctx: RunContext,
    llm_runtime: LLMRuntimeState,
) -> ReportDocument:
    """Optionally run the LLM report-polish flow and record warnings/fallbacks.

    Returns:
        ReportDocument: The final report.
    """
    if llm_processor is None:
        return report

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
                report, progress_callback=on_section_progress
            )
        except LLMProcessingError as error:
            _record_llm_fallback(
                ctx=ctx,
                st=st,
                llm_runtime=llm_runtime,
                error=error,
                elapsed_sec=st.elapsed_sec(),
            )
            return report
        section_elapsed_sec = st.elapsed_sec()
        st.set_detail(count_label(polish_plan.section_count, "section"))
    for warning in section_result.warnings:
        ctx.record_warning(warning)

    metadata_error: LLMProcessingError | None = None
    with stage(ctx, "llm_report_metadata", summary_label) as st:
        try:
            metadata_result = llm_processor.polish_report_metadata(
                report, section_transcripts=section_result.section_transcripts
            )
        except LLMProcessingError as error:
            metadata_elapsed_sec = st.elapsed_sec()
            _record_llm_fallback(
                ctx=ctx,
                st=st,
                llm_runtime=llm_runtime,
                error=error,
                elapsed_sec=section_elapsed_sec + metadata_elapsed_sec,
            )
            metadata_error = error
        else:
            metadata_elapsed_sec = st.elapsed_sec()
            st.set_detail(
                llm_report_detail(
                    section_count=polish_plan.section_count,
                    tldr_count=len(section_result.section_tldrs),
                    title_count=len(metadata_result.section_titles),
                    summary_count=len(metadata_result.summary),
                    action_item_count=len(metadata_result.action_items),
                )
            )

    if metadata_error is not None:
        llm_runtime.response_metadata = section_result.response_metadata
        return report

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
    llm_runtime.response_metadata = [
        *section_result.response_metadata,
        *metadata_result.response_metadata,
    ]
    return report


def _record_llm_fallback(
    *,
    ctx: RunContext,
    st: StageHandle,
    llm_runtime: LLMRuntimeState,
    error: LLMProcessingError,
    elapsed_sec: float,
) -> None:
    ctx.record_warning(str(error))
    st.set_detail(
        llm_fallback_detail(
            provider_name=llm_runtime.provider_name, model_name=llm_runtime.model_name
        )
    )
    llm_runtime.report_status = "fallback"
    llm_runtime.report_latency_sec = elapsed_sec
