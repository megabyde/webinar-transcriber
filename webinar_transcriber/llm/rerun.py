"""LLM-only report regeneration from a completed run."""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from webinar_transcriber.diagnostics import write_run_diagnostics
from webinar_transcriber.models import Diagnostics, LlmRerunDiagnostics, ReportDocument
from webinar_transcriber.paths import RunLayout
from webinar_transcriber.processor import RunContext, export_report, polish_report

if TYPE_CHECKING:
    from webinar_transcriber.llm.processor import InstructorLLMProcessor
    from webinar_transcriber.ui import StageReporter


class LlmRerunError(ValueError):
    """Raised when an existing run cannot safely supply an LLM rerun."""


@dataclass(slots=True, frozen=True)
class LlmRerunSource:
    """Validated deterministic report input and its provenance."""

    run_dir: Path
    report_path: Path
    report: ReportDocument
    report_sha256: str
    version: str


@dataclass(slots=True, frozen=True)
class LlmRerunArtifacts:
    """Artifacts returned from an LLM-only report rerun."""

    layout: RunLayout
    report: ReportDocument
    diagnostics: Diagnostics
    source_report_path: Path


def rerun_llm_report(
    source: LlmRerunSource,
    *,
    llm_processor: InstructorLLMProcessor,
    reporter: StageReporter,
) -> LlmRerunArtifacts:
    """Polish a completed run's deterministic report into a new variant.

    Returns:
        LlmRerunArtifacts: The new report variant and its diagnostics.
    """
    layout = _create_rerun_layout(source.run_dir)
    report = _relativize_image_paths(source.report, source.run_dir, layout.run_dir)
    ctx = RunContext(
        reporter=reporter,
        config=None,
        llm_rerun=LlmRerunDiagnostics(
            source_run=_relative_path(source.run_dir, layout.run_dir),
            source_report=_relative_path(source.report_path, layout.run_dir),
            source_report_sha256=source.report_sha256,
            source_version=source.version,
        ),
    )
    reporter.begin_llm_rerun(source.run_dir)
    try:
        report = polish_report(report, llm_processor=llm_processor, ctx=ctx)
        ctx.item_counts["report_sections"] = len(report.sections)
        export_report(report, layout=layout, ctx=ctx)
        diagnostics = write_run_diagnostics(layout, ctx, status="succeeded")
        artifacts = LlmRerunArtifacts(
            layout=layout,
            report=report,
            diagnostics=diagnostics,
            source_report_path=source.report_path,
        )
        reporter.complete_llm_rerun(artifacts)
        return artifacts
    except BaseException as ex:
        with suppress(Exception):
            write_run_diagnostics(
                layout,
                ctx,
                status="failed",
                failed_stage=ctx.failed_stage,
                error="Interrupted by user." if isinstance(ex, KeyboardInterrupt) else str(ex),
            )
        raise


def load_llm_rerun_source(run_dir: Path) -> LlmRerunSource:
    """Load and validate the deterministic report input from a completed run.

    Returns:
        LlmRerunSource: The validated report and source provenance.
    """
    resolved_run_dir = run_dir.resolve()
    diagnostics_path = _source_artifact_path(resolved_run_dir, "diagnostics.json")
    diagnostics = _read_json(diagnostics_path)
    if not isinstance(diagnostics, dict):
        raise LlmRerunError("diagnostics.json must be an object.")
    if diagnostics.get("llm_rerun") is not None:
        raise LlmRerunError(
            "The selected run directory is an LLM rerun variant. Pass its original source run "
            "directory instead."
        )
    if diagnostics.get("status") != "succeeded":
        raise LlmRerunError("LLM rerun requires a successfully completed source run.")

    version = _string(diagnostics.get("version"), "diagnostics.json.version")
    config = diagnostics.get("config")
    if not isinstance(config, dict):
        raise LlmRerunError("diagnostics.json.config must be an object.")
    used_llm = _boolean(config.get("llm"), "diagnostics.json.config.llm")

    local_report_path = _source_artifact_path(resolved_run_dir, "report.local.json")
    if local_report_path.exists():
        report_path = local_report_path
    elif used_llm:
        raise LlmRerunError(
            "Source run has no report.local.json. Older LLM-polished runs cannot be rerun without "
            "polishing LLM output twice."
        )
    else:
        report_path = _source_artifact_path(resolved_run_dir, "report.json")

    report_bytes = _read_bytes(report_path)
    report = _load_report(report_bytes, artifact=report_path.name, run_dir=resolved_run_dir)
    return LlmRerunSource(
        run_dir=resolved_run_dir,
        report_path=report_path,
        report=report,
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        version=version,
    )


def _create_rerun_layout(source_run_dir: Path) -> RunLayout:
    timestamp = datetime.now(tz=UTC).astimezone().strftime("%Y%m%d-%H%M%S-%f")
    variants_dir = source_run_dir / "llm"
    try:
        variants_dir.mkdir(exist_ok=True)
    except OSError as ex:
        raise LlmRerunError(f"Could not create LLM rerun directory: {ex}") from ex
    resolved_variants_dir = variants_dir.resolve()
    if not resolved_variants_dir.is_relative_to(source_run_dir):
        raise LlmRerunError("LLM rerun directory must stay inside the source run.")
    run_dir = resolved_variants_dir / timestamp
    try:
        run_dir.mkdir()
    except OSError as ex:
        raise LlmRerunError(f"Could not create LLM rerun directory: {ex}") from ex
    return RunLayout(run_dir=run_dir)


def _load_report(content: bytes, *, artifact: str, run_dir: Path) -> ReportDocument:
    # Pydantic belongs to the optional llm extra, so keep base CLI imports dependency-free
    from pydantic import TypeAdapter, ValidationError  # noqa: PLC0415

    try:
        report = TypeAdapter(ReportDocument).validate_json(content, strict=True)
    except ValidationError as ex:
        if ex.errors(include_url=False)[0]["type"] == "json_invalid":
            raise LlmRerunError(f"Source run has invalid JSON in {artifact}: {ex}") from ex
        raise LlmRerunError(f"Source run has an invalid report schema in {artifact}: {ex}") from ex

    ids: set[str] = set()
    for section in report.sections:
        if section.id in ids:
            raise LlmRerunError(f"Report contains duplicate section id: {section.id}")
        ids.add(section.id)
        if (
            not math.isfinite(section.start_sec)
            or not math.isfinite(section.end_sec)
            or section.start_sec < 0
            or section.end_sec < section.start_sec
        ):
            raise LlmRerunError(
                f"Report section {section.id} has invalid bounds: "
                f"{section.start_sec:g}-{section.end_sec:g}"
            )
        _validate_image_path(section.image_path, run_dir)
    return report


def _validate_image_path(image_path: str | None, run_dir: Path) -> None:
    if image_path is None:
        return
    relative_path = PurePosixPath(image_path)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or not relative_path.parts
        or relative_path.parts[0] != "frames"
    ):
        raise LlmRerunError(f"Report image path must stay under frames/: {image_path}")
    resolved_path = run_dir.joinpath(*relative_path.parts).resolve()
    if not resolved_path.is_relative_to(run_dir):
        raise LlmRerunError(f"Report image path must stay under frames/: {image_path}")


def _relativize_image_paths(
    report: ReportDocument, source_run_dir: Path, output_dir: Path
) -> ReportDocument:
    return replace(
        report,
        sections=[
            replace(
                section,
                image_path=(
                    _relative_path(source_run_dir / section.image_path, output_dir)
                    if section.image_path
                    else None
                ),
            )
            for section in report.sections
        ],
    )


def _relative_path(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start=start)).as_posix()


def _source_artifact_path(run_dir: Path, name: str) -> Path:
    path = (run_dir / name).resolve()
    if not path.is_relative_to(run_dir):
        raise LlmRerunError(f"Source artifact must stay inside the run directory: {name}")
    return path


def _read_json(path: Path) -> object:
    return _decode_json(_read_bytes(path), path.name)


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as ex:
        raise LlmRerunError(f"Source run is missing required artifact: {path.name}") from ex
    except OSError as ex:
        raise LlmRerunError(f"Could not read source artifact {path.name}: {ex}") from ex


def _decode_json(content: bytes, artifact: str) -> object:
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise LlmRerunError(f"Source run has invalid JSON in {artifact}: {ex}") from ex


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LlmRerunError(f"{label} must be a string.")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise LlmRerunError(f"{label} must be a boolean.")
    return value


__all__ = [
    "LlmRerunArtifacts",
    "LlmRerunError",
    "LlmRerunSource",
    "load_llm_rerun_source",
    "rerun_llm_report",
]
