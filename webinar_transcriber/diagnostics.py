"""Run-diagnostics assembly and persistence helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Literal

from webinar_transcriber.io import write_json
from webinar_transcriber.models import Diagnostics

if TYPE_CHECKING:
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.processor import RunContext

_ZERO_ITEM_COUNTS = {
    "transcript_segments": 0,
    "normalized_transcript_segments": 0,
    "vad_regions": 0,
    "windows": 0,
    "report_sections": 0,
    "scenes": 0,
    "frames": 0,
}


def write_run_diagnostics(
    layout: RunLayout,
    ctx: RunContext,
    *,
    status: Literal["succeeded", "failed"],
    failed_stage: str | None = None,
    error: str | None = None,
) -> Diagnostics:
    """Write diagnostics recorded on the processor context.

    Returns:
        Diagnostics: The assembled diagnostics payload.
    """
    diagnostics = Diagnostics(
        status=status,
        failed_stage=failed_stage,
        error=error,
        llm=ctx.llm,
        stage_durations_sec={key: round(value, 6) for key, value in ctx.stage_timings.items()},
        item_counts=_ZERO_ITEM_COUNTS | ctx.item_counts,
        asr_pipeline=ctx.asr_pipeline,
        diarization=ctx.diarization,
        warnings=ctx.warnings,
    )
    write_json(layout.diagnostics_path, asdict(diagnostics))
    return diagnostics
