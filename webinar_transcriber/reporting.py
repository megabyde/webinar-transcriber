"""Report polishing and export for initial runs and saved-report reruns."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from webinar_transcriber.export import write_docx_report, write_json_report, write_markdown_report
from webinar_transcriber.export.formatting import format_count, join_detail
from webinar_transcriber.llm import LlmProcessingError
from webinar_transcriber.models import (
    LlmDiagnostics,
)

if TYPE_CHECKING:
    from webinar_transcriber.diagnostics import RunContext
    from webinar_transcriber.llm import LlmReportMetadataResult
    from webinar_transcriber.llm.processor import InstructorLLMProcessor
    from webinar_transcriber.models import (
        ReportDocument,
        ReportStatus,
    )
    from webinar_transcriber.paths import RunLayout


def export_report(report: ReportDocument, *, layout: RunLayout, ctx: RunContext) -> None:
    """Write the Markdown, DOCX, and JSON artifacts for the report."""
    with ctx.stage("export", "Writing artifacts") as st:
        write_markdown_report(report, layout.markdown_report_path)
        write_docx_report(report, layout.docx_report_path)
        write_json_report(report, layout.json_report_path)
        st.update(detail="report.md | report.docx | report.json")


def polish_report(
    report: ReportDocument,
    *,
    llm_processor: InstructorLLMProcessor,
    ctx: RunContext,
) -> ReportDocument:
    provider_name = llm_processor.provider_name
    model_name = llm_processor.model_name
    section_count = len(report.sections)
    worker_count = llm_processor.polish_worker_count(section_count)
    runtime_detail = join_detail(provider_name, model_name)

    with ctx.stage(
        "llm_report_sections",
        "Polishing sections with LLM",
        total=section_count,
        detail=join_detail(runtime_detail, format_count(worker_count, "worker")),
    ) as st:

        def on_section_progress(advance: int) -> None:
            st.update(advance=advance)

        section_result = llm_processor.polish_report_sections(
            report, progress_callback=on_section_progress
        )
        section_elapsed_sec = st.elapsed_sec()
        st.update(detail=format_count(section_count, "section"))
    for warning in section_result.warnings:
        ctx.record_warning(warning)
    polished_report = replace(report, sections=section_result.sections)
    response_metadata = list(section_result.response_metadata)

    with ctx.stage(
        "llm_report_metadata", "Polishing report summary with LLM", detail=runtime_detail
    ) as st:
        try:
            metadata_result = llm_processor.polish_report_metadata(polished_report)
        except LlmProcessingError as ex:
            # The polished sections do not depend on the summary, so they are kept
            ctx.record_warning(str(ex))
            st.update(detail=join_detail(runtime_detail, "summary fallback"))
            metadata_result = None
        else:
            response_metadata.extend(metadata_result.response_metadata)
            st.update(detail=_metadata_detail(metadata_result))
        metadata_elapsed_sec = st.elapsed_sec()

    if metadata_result is not None:
        polished_report = replace(
            polished_report,
            summary=metadata_result.summary,
            action_items=metadata_result.action_items,
            sections=[
                replace(
                    section, title=metadata_result.section_titles.get(section.id, section.title)
                )
                for section in polished_report.sections
            ],
        )
    ctx.llm = LlmDiagnostics(
        provider=provider_name,
        model=model_name,
        report_status=_report_status(
            section_count=section_count,
            failed_section_count=section_result.failed_count,
            summary_applied=metadata_result is not None,
        ),
        report_latency_sec=section_elapsed_sec + metadata_elapsed_sec,
        response_metadata=response_metadata,
    )
    return polished_report


def _report_status(
    *, section_count: int, failed_section_count: int, summary_applied: bool
) -> ReportStatus:
    if summary_applied and failed_section_count == 0:
        return "applied"
    if not summary_applied and failed_section_count == section_count:
        return "fallback"
    return "partial"


def _metadata_detail(metadata_result: LlmReportMetadataResult) -> str:
    parts = []
    if metadata_result.summary:
        parts.append(format_count(len(metadata_result.summary), "summary bullet"))
    if metadata_result.action_items:
        parts.append(format_count(len(metadata_result.action_items), "action item"))
    if metadata_result.section_titles:
        parts.append(f"{format_count(len(metadata_result.section_titles), 'title')} updated")
    return join_detail(*parts)
