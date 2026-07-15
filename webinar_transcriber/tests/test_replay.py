"""Behavior-focused tests for report replay."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import ANY, Mock, patch

import pytest
from PIL import Image

from webinar_transcriber.io import write_json
from webinar_transcriber.llm import (
    LlmReportMetadataResult,
    LlmSectionPolishResult,
)
from webinar_transcriber.processor import process_replay
from webinar_transcriber.replay import ReplayValidationError, load_replay_source
from webinar_transcriber.tests.conftest import RecordingStageReporter

if TYPE_CHECKING:
    from pathlib import Path


def make_replay_source(
    tmp_path: Path, *, media_type: str = "video", diarization: bool = True
) -> Path:
    run_dir = tmp_path / "source-run"
    run_dir.mkdir()
    metadata: dict[str, object] = {
        "path": "/private/recordings/demo.mp4" if media_type == "video" else "demo.wav",
        "duration_sec": 12.0,
        "sample_rate": 16_000,
        "channels": 1,
        "media_type": media_type,
    }
    if media_type == "video":
        metadata.update({"fps": 30.0, "width": 1920, "height": 1080})
    write_json(run_dir / "metadata.json", metadata)
    write_json(
        run_dir / "transcript.json",
        {
            "detected_language": "en",
            "segments": [
                {
                    "id": "segment-1",
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "text": "Opening context.",
                    "speaker": "S1",
                },
                {
                    "id": "segment-2",
                    "start_sec": 6.0,
                    "end_sec": 11.0,
                    "text": "Next step send the notes.",
                    "speaker": "S2",
                },
            ],
        },
    )
    if media_type == "video":
        write_json(
            run_dir / "scenes.json",
            [
                {
                    "id": "scene-1",
                    "start_sec": 0.0,
                    "end_sec": 6.0,
                    "image_path": "frames/scene-1.png",
                },
                {
                    "id": "scene-2",
                    "start_sec": 6.0,
                    "end_sec": 12.0,
                    "image_path": "frames/scene-2.png",
                },
            ],
        )
        frames_dir = run_dir / "frames"
        frames_dir.mkdir()
        Image.new("RGB", (2, 2), color="red").save(frames_dir / "scene-1.png")
        Image.new("RGB", (2, 2), color="blue").save(frames_dir / "scene-2.png")
    if diarization:
        write_json(
            run_dir / "diarization.json",
            [
                {"start_sec": 0.0, "end_sec": 5.0, "speaker": "S1"},
                {"start_sec": 6.0, "end_sec": 11.0, "speaker": "S2"},
            ],
        )
    return run_dir


def snapshot_files(run_dir: Path) -> dict[str, bytes]:
    return {
        path.relative_to(run_dir).as_posix(): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }


class TestProcessReplay:
    def test_rebuilds_self_contained_video_report_without_upstream_stages(
        self, tmp_path: Path
    ) -> None:
        source_run = make_replay_source(tmp_path)
        source_before = snapshot_files(source_run)
        output_dir = tmp_path / "replay-run"
        reporter = RecordingStageReporter()

        with (
            patch("webinar_transcriber.processor.probe_media", side_effect=AssertionError),
            patch(
                "webinar_transcriber.processor.write_transcription_audio",
                side_effect=AssertionError,
            ),
            patch("webinar_transcriber.processor._run_asr_pipeline", side_effect=AssertionError),
            patch("webinar_transcriber.processor._detect_video_scenes", side_effect=AssertionError),
        ):
            artifacts = process_replay(source_run, output_dir=output_dir, reporter=reporter)

        assert reporter.completed is artifacts
        assert snapshot_files(source_run) == source_before
        for relative_path in (
            "metadata.json",
            "transcript.json",
            "scenes.json",
            "diarization.json",
            "frames/scene-1.png",
            "frames/scene-2.png",
        ):
            assert (output_dir / relative_path).read_bytes() == source_before[relative_path]
        assert artifacts.report.title == "Demo"
        assert [section.id for section in artifacts.report.sections] == [
            "section-1",
            "section-2",
        ]
        assert artifacts.report.sections[1].transcript_text == "**S2:** Next step send the notes."
        assert artifacts.diagnostics.mode == "replay"
        assert artifacts.diagnostics.replay is not None
        assert artifacts.diagnostics.replay.source_run == str(source_run.resolve())
        assert artifacts.diagnostics.asr_pipeline is None
        assert artifacts.diagnostics.diarization is None
        assert set(artifacts.diagnostics.stage_durations_sec) == {
            "validate_replay_source",
            "copy_replay_artifacts",
            "export",
        }
        assert artifacts.diagnostics.item_counts == {
            "transcript_segments": 2,
            "coalesced_transcript_segments": 2,
            "scenes": 2,
            "frames": 2,
            "report_sections": 2,
        }
        diagnostics = json.loads(artifacts.layout.diagnostics_path.read_text(encoding="utf-8"))
        assert diagnostics["mode"] == "replay"
        assert diagnostics["replay"] == {"source_run": str(source_run.resolve())}
        assert not (output_dir / "asr").exists()
        assert not (output_dir / "transcription-audio.mp3").exists()

    def test_replay_is_deterministic_for_the_same_persisted_inputs(self, tmp_path: Path) -> None:
        source_run = make_replay_source(tmp_path)

        first = process_replay(
            source_run, output_dir=tmp_path / "first", reporter=RecordingStageReporter()
        )
        second = process_replay(
            source_run, output_dir=tmp_path / "second", reporter=RecordingStageReporter()
        )

        assert first.layout.markdown_report_path.read_bytes() == (
            second.layout.markdown_report_path.read_bytes()
        )
        assert (
            first.layout.json_report_path.read_bytes()
            == second.layout.json_report_path.read_bytes()
        )
        assert first.report == second.report

    def test_replays_audio_without_video_or_diarization_artifacts(self, tmp_path: Path) -> None:
        source_run = make_replay_source(tmp_path, media_type="audio", diarization=False)

        artifacts = process_replay(
            source_run, output_dir=tmp_path / "audio-replay", reporter=RecordingStageReporter()
        )

        assert artifacts.media_asset.media_type == "audio"
        assert len(artifacts.report.sections) == 1
        assert not artifacts.layout.scenes_path.exists()
        assert not artifacts.layout.frames_dir.exists()
        assert not artifacts.layout.diarization_path.exists()

    def test_replay_applies_optional_llm_refinement(self, tmp_path: Path) -> None:
        source_run = make_replay_source(tmp_path)
        llm = Mock(provider_name="openai", model_name="replay-model")
        llm.polish_worker_count.return_value = 2
        llm.polish_report_sections.return_value = LlmSectionPolishResult(
            section_tldrs={"section-1": "Opening recap."},
            section_transcripts={"section-1": "Polished opening."},
            warnings=["section-2 stayed local"],
        )
        llm.polish_report_metadata.return_value = LlmReportMetadataResult(
            summary=["Summary"],
            action_items=["Send notes"],
            section_titles={"section-1": "Opening"},
        )

        artifacts = process_replay(
            source_run,
            output_dir=tmp_path / "llm-replay",
            reporter=RecordingStageReporter(),
            llm_processor=llm,
        )

        llm.polish_report_metadata.assert_called_once_with(
            ANY,
            section_transcripts={"section-1": "Polished opening."},
        )
        assert artifacts.report.summary == ["Summary"]
        assert artifacts.report.sections[0].title == "Opening"
        assert artifacts.report.sections[0].transcript_text == "Polished opening."
        assert artifacts.diagnostics.warnings == ["section-2 stayed local"]
        assert artifacts.diagnostics.llm is not None
        assert artifacts.diagnostics.llm.model == "replay-model"
        assert artifacts.diagnostics.llm.report_status == "applied"

    def test_writes_replay_diagnostics_when_report_generation_fails(self, tmp_path: Path) -> None:
        source_run = make_replay_source(tmp_path)
        output_dir = tmp_path / "failed-replay"

        with (
            patch(
                "webinar_transcriber.processor.write_markdown_report",
                side_effect=OSError("report write failed"),
            ),
            pytest.raises(OSError, match="report write failed"),
        ):
            process_replay(source_run, output_dir=output_dir, reporter=RecordingStageReporter())

        diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
        assert diagnostics["status"] == "failed"
        assert diagnostics["failed_stage"] == "export"
        assert diagnostics["mode"] == "replay"

    def test_maps_copy_errors_to_replay_validation_errors(self, tmp_path: Path) -> None:
        source_run = make_replay_source(tmp_path)

        with (
            patch(
                "webinar_transcriber.replay.shutil.copy2",
                side_effect=FileNotFoundError("source artifact disappeared"),
            ),
            pytest.raises(
                ReplayValidationError,
                match="Could not copy replay artifacts: source artifact disappeared",
            ),
        ):
            process_replay(
                source_run,
                output_dir=tmp_path / "failed-copy",
                reporter=RecordingStageReporter(),
            )


class TestReplayValidation:
    def test_accepts_unknown_media_duration_with_timed_artifacts(self, tmp_path: Path) -> None:
        source_run = make_replay_source(tmp_path)
        metadata_path = source_run / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["duration_sec"] = 0.0
        write_json(metadata_path, metadata)

        source = load_replay_source(source_run)

        assert source.media_asset.duration_sec == 0.0
        assert source.transcription.segments[-1].end_sec == 11.0
        assert source.scenes[-1].end_sec == 12.0

    def test_accepts_timeline_rounding_beyond_header_duration(self, tmp_path: Path) -> None:
        source_run = make_replay_source(tmp_path, media_type="audio", diarization=False)
        transcript_path = source_run / "transcript.json"
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        transcript["segments"][-1]["end_sec"] = 12.5
        write_json(transcript_path, transcript)

        source = load_replay_source(source_run)

        assert source.transcription.segments[-1].end_sec == 12.5

    def test_accepts_null_optional_metadata_and_transcript_fields(self, tmp_path: Path) -> None:
        source_run = make_replay_source(tmp_path)
        metadata_path = source_run / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({
            "sample_rate": None,
            "channels": None,
            "fps": None,
            "width": None,
            "height": None,
        })
        write_json(metadata_path, metadata)
        transcript_path = source_run / "transcript.json"
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        transcript["detected_language"] = None
        for segment in transcript["segments"]:
            segment.pop("speaker")
        write_json(transcript_path, transcript)

        source = load_replay_source(source_run)

        assert source.media_asset.sample_rate is None
        assert source.transcription.detected_language is None
        assert all(segment.speaker is None for segment in source.transcription.segments)

    def test_rejects_missing_required_artifact_before_creating_destination(
        self, tmp_path: Path
    ) -> None:
        source_run = tmp_path / "source-run"
        source_run.mkdir()
        output_dir = tmp_path / "destination"

        with pytest.raises(
            ReplayValidationError, match=r"missing required artifact: metadata\.json"
        ):
            process_replay(source_run, output_dir=output_dir, reporter=RecordingStageReporter())

        assert not output_dir.exists()

    @pytest.mark.parametrize(
        ("artifact", "payload", "message"),
        [
            ("metadata.json", "not json", "invalid JSON in metadata.json"),
            ("metadata.json", [], "metadata.json must be an object"),
            (
                "metadata.json",
                {
                    "path": "demo.mp4",
                    "duration_sec": 12.0,
                    "sample_rate": 16_000,
                    "channels": 1,
                    "media_type": "document",
                },
                "media_type must be 'audio' or 'video'",
            ),
            (
                "metadata.json",
                {
                    "path": "demo.mp4",
                    "duration_sec": 12.0,
                    "sample_rate": 16_000,
                    "channels": 1,
                    "media_type": "video",
                    "fps": 30.0,
                    "width": 1920,
                },
                "missing keys: height",
            ),
        ],
    )
    def test_rejects_invalid_top_level_artifacts(
        self, tmp_path: Path, artifact: str, payload: object, message: str
    ) -> None:
        source_run = make_replay_source(tmp_path)
        path = source_run / artifact
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            write_json(path, payload)

        with pytest.raises(ReplayValidationError, match=message):
            load_replay_source(source_run)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda data: data.update({"extra": True}), "unknown keys: extra"),
            (lambda data: data.update({"path": ""}), "metadata.json.path must be a string"),
            (lambda data: data.update({"duration_sec": True}), "duration_sec must be a number"),
            (
                lambda data: data.update({"duration_sec": -1}),
                "duration_sec must be nonnegative",
            ),
            (
                lambda data: data.update({"sample_rate": 1.5}),
                "sample_rate must be a positive integer",
            ),
            (lambda data: data.update({"fps": 0}), "metadata.json.fps must be positive"),
        ],
    )
    def test_rejects_invalid_metadata_fields(self, tmp_path: Path, mutate, message: str) -> None:
        source_run = make_replay_source(tmp_path)
        path = source_run / "metadata.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        mutate(data)
        write_json(path, data)

        with pytest.raises(ReplayValidationError, match=message):
            load_replay_source(source_run)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (
                lambda data: data.update({"detected_language": 3}),
                "detected_language must be a string",
            ),
            (lambda data: data.update({"segments": {}}), "segments must be an array"),
            (
                lambda data: data["segments"][1].update({"id": "segment-1"}),
                "duplicate segment id: segment-1",
            ),
            (
                lambda data: data["segments"][0].update({"end_sec": 13.1}),
                "invalid bounds: 0-13.1",
            ),
            (
                lambda data: data["segments"][0].update({"text": 4}),
                "segments\\[0\\].text must be a string",
            ),
        ],
    )
    def test_rejects_invalid_transcript_fields(self, tmp_path: Path, mutate, message: str) -> None:
        source_run = make_replay_source(tmp_path)
        path = source_run / "transcript.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        mutate(data)
        write_json(path, data)

        with pytest.raises(ReplayValidationError, match=message):
            load_replay_source(source_run)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (
                lambda data: data[1].update({"id": "scene-1"}),
                "duplicate scene id: scene-1",
            ),
            (
                lambda data: data[0].update({"image_path": "../secret.png"}),
                "frame path must stay under frames/",
            ),
            (
                lambda data: data[1].update({"image_path": "frames/scene-1.png"}),
                "duplicate frame path: frames/scene-1.png",
            ),
            (
                lambda data: data[0].update({"image_path": "frames/missing.png"}),
                "missing referenced frame: frames/missing.png",
            ),
        ],
    )
    def test_rejects_invalid_scene_and_frame_references(
        self, tmp_path: Path, mutate, message: str
    ) -> None:
        source_run = make_replay_source(tmp_path)
        path = source_run / "scenes.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        mutate(data)
        write_json(path, data)

        with pytest.raises(ReplayValidationError, match=message):
            load_replay_source(source_run)

    def test_rejects_invalid_optional_diarization(self, tmp_path: Path) -> None:
        source_run = make_replay_source(tmp_path)
        write_json(
            source_run / "diarization.json",
            [{"start_sec": 0.0, "end_sec": 5.0, "speaker": None}],
        )

        with pytest.raises(ReplayValidationError, match="speaker must be a string"):
            load_replay_source(source_run)

    def test_rejects_frame_symlink_outside_source_run(self, tmp_path: Path) -> None:
        source_run = make_replay_source(tmp_path)
        external_frame = tmp_path / "external.png"
        Image.new("RGB", (2, 2)).save(external_frame)
        source_frame = source_run / "frames" / "scene-1.png"
        source_frame.unlink()
        source_frame.symlink_to(external_frame)

        with pytest.raises(ReplayValidationError, match="frame path must stay under frames/"):
            load_replay_source(source_run)
