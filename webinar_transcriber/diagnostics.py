"""Run-diagnostics assembly and persistence helpers."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from webinar_transcriber.io import write_json
from webinar_transcriber.models import Diagnostics

if TYPE_CHECKING:
    from collections.abc import Generator

    from webinar_transcriber.models import (
        AsrPipelineDiagnostics,
        DiarizationDiagnostics,
        LlmDiagnostics,
        LlmRerunDiagnostics,
        RunConfig,
    )
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.ui import StageHandle, StageReporter


@dataclass
class RunContext:
    """Mutable state and recorded diagnostics for one processing run."""

    reporter: StageReporter
    config: RunConfig | None
    stage_timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failed_stage: str | None = None
    item_counts: dict[str, int] = field(default_factory=dict)
    asr_pipeline: AsrPipelineDiagnostics | None = None
    diarization: DiarizationDiagnostics | None = None
    llm: LlmDiagnostics | None = None
    llm_rerun: LlmRerunDiagnostics | None = None

    def record_warning(self, message: str) -> None:
        """Record and report one run warning."""
        self.warnings.append(message)
        self.reporter.warn(message)

    @contextmanager
    def stage(
        self, key: str, label: str, *, total: float | None = None, detail: str | None = None
    ) -> Generator[StageHandle, None, None]:
        """Track timing for one stage and surface it through the reporter."""
        with self.reporter.track(key, label, total=total, detail=detail) as handle:
            try:
                yield handle
            except BaseException:
                self.failed_stage = key
                raise
            finally:
                self.stage_timings[key] = handle.elapsed_sec()

    @contextmanager
    def failure_diagnostics(self, layout: RunLayout) -> Generator[None, None, None]:
        """Write failed-run diagnostics when the wrapped block raises, then re-raise."""
        try:
            yield
        except BaseException as ex:
            # A diagnostics failure must not mask the original error
            with suppress(Exception):
                write_run_diagnostics(
                    layout,
                    self,
                    status="failed",
                    error="Interrupted by user." if isinstance(ex, KeyboardInterrupt) else str(ex),
                )
            raise


def write_run_diagnostics(
    layout: RunLayout,
    ctx: RunContext,
    *,
    status: Literal["succeeded", "failed"],
    error: str | None = None,
) -> Diagnostics:
    """Write diagnostics recorded on the processor context.

    Returns:
        Diagnostics: The assembled diagnostics payload.
    """
    diagnostics = Diagnostics(
        status=status,
        failed_stage=ctx.failed_stage,
        error=error,
        config=ctx.config,
        llm=ctx.llm,
        llm_rerun=ctx.llm_rerun,
        stage_durations_sec={key: round(value, 6) for key, value in ctx.stage_timings.items()},
        item_counts=dict(ctx.item_counts),
        asr_pipeline=ctx.asr_pipeline,
        diarization=ctx.diarization,
        warnings=ctx.warnings,
    )
    write_json(layout.diagnostics_path, asdict(diagnostics))
    return diagnostics
