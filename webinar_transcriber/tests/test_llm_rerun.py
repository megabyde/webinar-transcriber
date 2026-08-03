"""Tests for LLM-only report reruns."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from rich.console import Console

from webinar_transcriber import __version__
from webinar_transcriber.io import write_json
from webinar_transcriber.llm import LlmReportMetadataResult, LlmSectionPolishResult
from webinar_transcriber.llm.rerun import (
    LlmRerunError,
    load_llm_rerun_source,
    rerun_llm_report,
)
from webinar_transcriber.models import MediaType, ReportDocument, ReportSection
from webinar_transcriber.ui import StageReporter


def _report_payload(*, image_path: str | None = None) -> dict[str, object]:
    return ReportDocument(
        title="Local report",
        source_file="demo.mp4",
        media_type=MediaType.VIDEO,
        detected_language="en",
        sections=[
            ReportSection(
                id="section-1",
                start_sec=0.0,
                end_sec=10.0,
                title="Local section",
                transcript_text="Local transcript.",
                image_path=image_path,
            )
        ],
    ).to_json()


def _write_source_run(
    run_dir: Path,
    *,
    used_llm: bool = False,
    local_report: bool = False,
    image: bool = False,
) -> Path:
    run_dir.mkdir()
    write_json(
        run_dir / "diagnostics.json",
        {"version": "1.4.0", "status": "succeeded", "config": {"llm": used_llm}},
    )
    image_path = "frames/section-1.png" if image else None
    if image:
        frame_path = run_dir / "frames/section-1.png"
        frame_path.parent.mkdir()
        Image.new("RGB", (4, 4), color="white").save(frame_path)
    report_path = run_dir / ("report.local.json" if local_report else "report.json")
    write_json(report_path, _report_payload(image_path=image_path))
    return report_path


def _llm_processor() -> Mock:
    processor = Mock(provider_name="openai", model_name="gpt-test")
    processor.polish_worker_count.return_value = 1
    processor.polish_report_sections.return_value = LlmSectionPolishResult(
        section_tldrs={"section-1": "Polished TL;DR."},
        section_transcripts={"section-1": "Polished transcript."},
        response_metadata=[{"stage": "section_polish", "finish_reason": "stop"}],
        warnings=["Provider warning."],
    )
    processor.polish_report_metadata.return_value = LlmReportMetadataResult(
        summary=["Polished summary."],
        action_items=["Polished action."],
        section_titles={"section-1": "Polished section"},
        response_metadata=[{"stage": "metadata_polish", "finish_reason": "stop"}],
    )
    return processor


class TestRerunLlmReport:
    def test_writes_variant_with_provenance_without_changing_source(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "source-run"
        source_report = _write_source_run(run_dir, used_llm=True, local_report=True, image=True)
        write_json(run_dir / "report.json", {"original": "polished"})
        (run_dir / "report.md").write_text("original markdown", encoding="utf-8")
        (run_dir / "report.docx").write_bytes(b"original docx")
        source_artifacts = {
            path.name: path.read_bytes()
            for path in (
                source_report,
                run_dir / "report.json",
                run_dir / "report.md",
                run_dir / "report.docx",
                run_dir / "diagnostics.json",
            )
        }
        processor = _llm_processor()

        artifacts = rerun_llm_report(
            load_llm_rerun_source(run_dir),
            llm_processor=processor,
            reporter=StageReporter(console=Console(quiet=True)),
        )

        assert artifacts.layout.run_dir.parent == run_dir / "llm"
        assert artifacts.layout.markdown_report_path.exists()
        assert artifacts.layout.docx_report_path.exists()
        assert artifacts.layout.json_report_path.exists()
        assert artifacts.report.summary == ["Polished summary."]
        assert artifacts.report.sections[0].image_path == "../../frames/section-1.png"
        output_payload = json.loads(artifacts.layout.json_report_path.read_text(encoding="utf-8"))
        assert output_payload["sections"][0]["image_path"] == "../../frames/section-1.png"
        assert artifacts.diagnostics.version == __version__
        assert artifacts.diagnostics.status == "succeeded"
        assert artifacts.diagnostics.config is None
        assert artifacts.diagnostics.llm is not None
        assert artifacts.diagnostics.llm.provider == "openai"
        assert artifacts.diagnostics.llm.model == "gpt-test"
        assert artifacts.diagnostics.llm_rerun is not None
        assert artifacts.diagnostics.llm_rerun.source_run == "../.."
        assert artifacts.diagnostics.llm_rerun.source_report == "../../report.local.json"
        assert (
            artifacts.diagnostics.llm_rerun.source_report_sha256
            == hashlib.sha256(source_report.read_bytes()).hexdigest()
        )
        assert artifacts.diagnostics.llm_rerun.source_version == "1.4.0"
        diagnostics_payload = json.loads(
            artifacts.layout.diagnostics_path.read_text(encoding="utf-8")
        )
        assert diagnostics_payload["llm_rerun"] == {
            "source_run": "../..",
            "source_report": "../../report.local.json",
            "source_report_sha256": hashlib.sha256(source_report.read_bytes()).hexdigest(),
            "source_version": "1.4.0",
        }
        assert artifacts.diagnostics.item_counts == {"report_sections": 1}
        assert set(artifacts.diagnostics.stage_durations_sec) == {
            "llm_report_sections",
            "llm_report_metadata",
            "export",
        }
        assert artifacts.diagnostics.warnings == ["Provider warning."]
        assert {
            path.name: path.read_bytes()
            for path in (
                source_report,
                run_dir / "report.json",
                run_dir / "report.md",
                run_dir / "report.docx",
                run_dir / "diagnostics.json",
            )
        } == source_artifacts

    def test_uses_final_report_from_legacy_non_llm_run(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "source-run"
        report_path = _write_source_run(run_dir)

        source = load_llm_rerun_source(run_dir)

        assert source.report_path == report_path.resolve()
        assert source.report.title == "Local report"

    def test_records_failure_diagnostics_after_output_directory_exists(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "source-run"
        _write_source_run(run_dir)

        with (
            patch(
                "webinar_transcriber.processor.write_docx_report", side_effect=RuntimeError("boom")
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            rerun_llm_report(
                load_llm_rerun_source(run_dir),
                llm_processor=_llm_processor(),
                reporter=StageReporter(console=Console(quiet=True)),
            )

        output_dir = next((run_dir / "llm").iterdir())
        diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
        assert diagnostics["status"] == "failed"
        assert diagnostics["failed_stage"] == "export"
        assert diagnostics["error"] == "boom"
        assert diagnostics["llm_rerun"]["source_report"] == "../../report.json"

    def test_records_interruption_in_failure_diagnostics(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "source-run"
        _write_source_run(run_dir)

        with (
            patch("webinar_transcriber.llm.rerun.polish_report", side_effect=KeyboardInterrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            rerun_llm_report(
                load_llm_rerun_source(run_dir),
                llm_processor=_llm_processor(),
                reporter=StageReporter(console=Console(quiet=True)),
            )

        output_dir = next((run_dir / "llm").iterdir())
        diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
        assert diagnostics["status"] == "failed"
        assert diagnostics["error"] == "Interrupted by user."
        assert diagnostics["config"] is None
        assert diagnostics["llm"] is None
        assert diagnostics["llm_rerun"]["source_report"] == "../../report.json"

    def test_diagnostics_failure_does_not_mask_processing_failure(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "source-run"
        _write_source_run(run_dir)

        with (
            patch("webinar_transcriber.llm.rerun.polish_report", side_effect=RuntimeError("boom")),
            patch(
                "webinar_transcriber.llm.rerun.write_run_diagnostics",
                side_effect=OSError("disk full"),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            rerun_llm_report(
                load_llm_rerun_source(run_dir),
                llm_processor=_llm_processor(),
                reporter=StageReporter(console=Console(quiet=True)),
            )

    def test_reports_output_directory_creation_failure(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "source-run"
        _write_source_run(run_dir)

        with (
            patch.object(Path, "mkdir", side_effect=OSError("read-only")),
            pytest.raises(LlmRerunError, match="Could not create LLM rerun directory"),
        ):
            rerun_llm_report(
                load_llm_rerun_source(run_dir),
                llm_processor=_llm_processor(),
                reporter=StageReporter(console=Console(quiet=True)),
            )

    def test_reports_variant_directory_creation_failure(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "source-run"
        _write_source_run(run_dir)
        (run_dir / "llm").mkdir()

        def fail_variant(path: Path, **_kwargs) -> None:
            if path != run_dir / "llm":
                raise OSError("read-only")

        with (
            patch.object(Path, "mkdir", fail_variant),
            pytest.raises(LlmRerunError, match="Could not create LLM rerun directory"),
        ):
            rerun_llm_report(
                load_llm_rerun_source(run_dir),
                llm_processor=_llm_processor(),
                reporter=StageReporter(console=Console(quiet=True)),
            )

    def test_rejects_variant_directory_symlink_that_escapes_run(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "source-run"
        _write_source_run(run_dir)
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        (run_dir / "llm").symlink_to(outside_dir, target_is_directory=True)

        with pytest.raises(LlmRerunError, match="must stay inside the source run"):
            rerun_llm_report(
                load_llm_rerun_source(run_dir),
                llm_processor=_llm_processor(),
                reporter=StageReporter(console=Console(quiet=True)),
            )

        assert list(outside_dir.iterdir()) == []


class TestLoadLlmRerunSource:
    def test_rejects_unsuccessful_run(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "source-run"
        _write_source_run(run_dir)
        write_json(
            run_dir / "diagnostics.json",
            {"version": "1.4.0", "status": "failed", "config": {"llm": False}},
        )

        with pytest.raises(LlmRerunError, match="successfully completed"):
            load_llm_rerun_source(run_dir)

    def test_rejects_legacy_llm_run_without_local_report(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "source-run"
        _write_source_run(run_dir, used_llm=True)

        with pytest.raises(LlmRerunError, match="Older LLM-polished runs cannot be rerun"):
            load_llm_rerun_source(run_dir)

    def test_rejects_llm_rerun_variant_with_original_run_guidance(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "variant"
        _write_source_run(run_dir, used_llm=True)
        write_json(
            run_dir / "diagnostics.json",
            {
                "version": "1.5.0",
                "status": "failed",
                "config": None,
                "llm_rerun": {"source_run": "../.."},
            },
        )

        with pytest.raises(LlmRerunError, match=r"rerun variant.*original source run"):
            load_llm_rerun_source(run_dir)

    @pytest.mark.parametrize(
        ("artifact", "content", "message"),
        [
            ("diagnostics.json", b"{", "invalid JSON in diagnostics.json"),
            ("diagnostics.json", b"\xff", "invalid JSON in diagnostics.json"),
            ("report.json", b"{", "invalid JSON in report.json"),
        ],
    )
    def test_rejects_invalid_json(
        self, tmp_path: Path, artifact: str, content: bytes, message: str
    ) -> None:
        run_dir = tmp_path / "source-run"
        _write_source_run(run_dir)
        (run_dir / artifact).write_bytes(content)

        with pytest.raises(LlmRerunError, match=message):
            load_llm_rerun_source(run_dir)

    @pytest.mark.parametrize("artifact", ["diagnostics.json", "report.json"])
    def test_rejects_missing_required_artifact(self, tmp_path: Path, artifact: str) -> None:
        run_dir = tmp_path / "source-run"
        _write_source_run(run_dir)
        (run_dir / artifact).unlink()

        with pytest.raises(LlmRerunError, match=f"missing required artifact: {artifact}"):
            load_llm_rerun_source(run_dir)

    def test_reports_artifact_read_failure(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "source-run"
        _write_source_run(run_dir)
        original_read_bytes = Path.read_bytes

        def fail_report(path: Path) -> bytes:
            if path.name == "report.json":
                raise OSError("unreadable")
            return original_read_bytes(path)

        with (
            patch.object(Path, "read_bytes", fail_report),
            pytest.raises(LlmRerunError, match=r"Could not read source artifact report\.json"),
        ):
            load_llm_rerun_source(run_dir)

    def test_rejects_source_report_symlink_that_escapes_run(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "source-run"
        report_path = _write_source_run(run_dir)
        outside_report = tmp_path / "outside.json"
        write_json(outside_report, _report_payload())
        report_path.unlink()
        report_path.symlink_to(outside_report)

        with pytest.raises(LlmRerunError, match="Source artifact must stay inside"):
            load_llm_rerun_source(run_dir)

    @pytest.mark.parametrize(
        ("diagnostics", "message"),
        [
            ([], "diagnostics.json must be an object"),
            (
                {"version": 1, "status": "succeeded", "config": {"llm": False}},
                "diagnostics.json.version must be a string",
            ),
            (
                {"version": "1.4.0", "status": "succeeded", "config": []},
                "diagnostics.json.config must be an object",
            ),
            (
                {"version": "1.4.0", "status": "succeeded", "config": {"llm": "no"}},
                "diagnostics.json.config.llm must be a boolean",
            ),
        ],
    )
    def test_rejects_invalid_diagnostics(
        self, tmp_path: Path, diagnostics: object, message: str
    ) -> None:
        run_dir = tmp_path / "source-run"
        _write_source_run(run_dir)
        write_json(run_dir / "diagnostics.json", diagnostics)

        with pytest.raises(LlmRerunError, match=message):
            load_llm_rerun_source(run_dir)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda _payload: [], "invalid report schema"),
            (
                lambda payload: {key: value for key, value in payload.items() if key != "title"},
                "invalid report schema",
            ),
            (
                lambda payload: {**payload, "summary": "summary"},
                "invalid report schema",
            ),
            (
                lambda payload: {
                    **payload,
                    "sections": [
                        payload["sections"][0],
                        {**payload["sections"][0], "title": "Duplicate"},
                    ],
                },
                "duplicate section id",
            ),
            (
                lambda payload: {
                    **payload,
                    "sections": [{**payload["sections"][0], "start_sec": True}],
                },
                "invalid report schema",
            ),
            (
                lambda payload: {
                    **payload,
                    "sections": [{**payload["sections"][0], "end_sec": float("inf")}],
                },
                "invalid bounds",
            ),
            (
                lambda payload: {
                    **payload,
                    "sections": [{**payload["sections"][0], "start_sec": -1}],
                },
                "invalid bounds",
            ),
        ],
    )
    def test_rejects_invalid_report_schema(self, tmp_path: Path, mutate, message: str) -> None:
        run_dir = tmp_path / "source-run"
        report_path = _write_source_run(run_dir)
        payload = mutate(_report_payload())
        write_json(report_path, payload)

        with pytest.raises(LlmRerunError, match=message):
            load_llm_rerun_source(run_dir)

    @pytest.mark.parametrize(
        ("image_path", "message"),
        [
            ("/frame.png", "must stay under frames"),
            ("frames/../outside.png", "must stay under frames"),
            ("other/frame.png", "must stay under frames"),
        ],
    )
    def test_rejects_unsafe_frame(self, tmp_path: Path, image_path: str, message: str) -> None:
        run_dir = tmp_path / "source-run"
        report_path = _write_source_run(run_dir)
        write_json(report_path, _report_payload(image_path=image_path))

        with pytest.raises(LlmRerunError, match=message):
            load_llm_rerun_source(run_dir)

    def test_allows_missing_frame_for_exporter_fallback(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "source-run"
        report_path = _write_source_run(run_dir)
        write_json(report_path, _report_payload(image_path="frames/missing.png"))

        source = load_llm_rerun_source(run_dir)

        assert source.report.sections[0].image_path == "frames/missing.png"

    def test_rejects_frame_symlink_that_escapes_run(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "source-run"
        report_path = _write_source_run(run_dir)
        outside_frame = tmp_path / "outside.png"
        Image.new("RGB", (4, 4), color="white").save(outside_frame)
        (run_dir / "frames").mkdir()
        (run_dir / "frames" / "linked.png").symlink_to(outside_frame)
        write_json(report_path, _report_payload(image_path="frames/linked.png"))

        with pytest.raises(LlmRerunError, match="must stay under frames"):
            load_llm_rerun_source(run_dir)
