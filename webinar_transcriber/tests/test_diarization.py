"""Tests for local speaker-diarization helpers."""

from __future__ import annotations

import hashlib
import tarfile
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Self
from unittest.mock import Mock

import numpy as np
import pytest

import webinar_transcriber.diarization.sherpa_diarizer as sherpa_runtime
from webinar_transcriber.diarization import assign_speakers
from webinar_transcriber.diarization.contracts import DiarizationProcessingError
from webinar_transcriber.diarization.sherpa_diarizer import (
    SEGMENTATION_MODEL,
    default_cache_dir,
    default_model_paths,
    normalize_speaker_labels,
)
from webinar_transcriber.models import SpeakerTurn, TranscriptSegment


def _segment(
    segment_id: str, start_sec: float, end_sec: float, *, text: str = "text"
) -> TranscriptSegment:
    return TranscriptSegment(id=segment_id, start_sec=start_sec, end_sec=end_sec, text=text)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        if size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class FakeSherpaItem:
    def __init__(self, start: float, end: float, speaker: int) -> None:
        self.start = start
        self.end = end
        self.speaker = speaker


class FakeSherpaResult:
    def __init__(self, items: list[FakeSherpaItem]) -> None:
        self._items = items

    def sort_by_start_time(self) -> list[FakeSherpaItem]:
        return sorted(self._items, key=lambda item: item.start)


class FakeSherpaConfig:
    def __init__(self, *, valid: bool) -> None:
        self._valid = valid

    def validate(self) -> bool:
        return self._valid


class FakeSherpaModule(ModuleType):
    def __init__(
        self,
        *,
        results: list[list[FakeSherpaItem]] | None = None,
        valid: bool = True,
        runtime_error: RuntimeError | None = None,
    ) -> None:
        super().__init__("fake_sherpa_onnx")
        self.results = list(results or [[]])
        self.valid = valid
        self.runtime_error = runtime_error
        self.cluster_counts: list[int] = []
        self.thread_counts: list[int] = []

    def OfflineSpeakerSegmentationPyannoteModelConfig(self, *, model: str):  # noqa: N802
        return SimpleNamespace(model=model)

    def OfflineSpeakerSegmentationModelConfig(  # noqa: N802
        self, *, pyannote, num_threads: int = 1
    ):
        self.thread_counts.append(num_threads)
        return SimpleNamespace(pyannote=pyannote)

    def SpeakerEmbeddingExtractorConfig(self, *, model: str, num_threads: int = 1):  # noqa: N802
        self.thread_counts.append(num_threads)
        return SimpleNamespace(model=model)

    def FastClusteringConfig(self, *, num_clusters: int, threshold: float):  # noqa: N802
        self.cluster_counts.append(num_clusters)
        return SimpleNamespace(num_clusters=num_clusters, threshold=threshold)

    def OfflineSpeakerDiarizationConfig(self, **_kwargs):  # noqa: N802
        return FakeSherpaConfig(valid=self.valid)

    def OfflineSpeakerDiarization(self, _config):  # noqa: N802
        module = self

        class FakeDiarization:
            def process(self, _samples, *, callback=None):
                if module.runtime_error is not None:
                    raise module.runtime_error
                if callback is not None:
                    callback(1, 2)
                    callback(2, 2)
                return FakeSherpaResult(module.results.pop(0))

        return FakeDiarization()


class TestAssignSpeakers:
    def test_assigns_by_largest_temporal_overlap(self) -> None:
        segments = [_segment("segment-1", 1.0, 5.0)]
        turns = [
            SpeakerTurn(start_sec=0.0, end_sec=2.0, speaker="S1"),
            SpeakerTurn(start_sec=2.0, end_sec=5.0, speaker="S2"),
        ]

        assigned = assign_speakers(segments, turns)

        assert assigned[0].speaker == "S2"
        assert segments[0].speaker is None

    def test_leaves_segments_without_overlap_unknown(self) -> None:
        assigned = assign_speakers(
            [_segment("segment-1", 1.0, 2.0)],
            [
                SpeakerTurn(start_sec=0.0, end_sec=1.0, speaker="S1"),
                SpeakerTurn(start_sec=1.5, end_sec=1.5, speaker="S2"),
            ],
        )

        assert assigned[0].speaker is None

    def test_tie_breaks_by_closer_midpoint(self) -> None:
        assigned = assign_speakers(
            [_segment("segment-1", 2.0, 4.0)],
            [
                SpeakerTurn(start_sec=0.0, end_sec=3.0, speaker="S1"),
                SpeakerTurn(start_sec=3.0, end_sec=4.5, speaker="S2"),
            ],
        )

        assert assigned[0].speaker == "S2"

    def test_preserves_input_segment_order(self) -> None:
        assigned = assign_speakers(
            [_segment("segment-2", 10.0, 11.0), _segment("segment-1", 1.0, 2.0)],
            [
                SpeakerTurn(start_sec=0.0, end_sec=3.0, speaker="S1"),
                SpeakerTurn(start_sec=9.0, end_sec=12.0, speaker="S2"),
            ],
        )

        assert [(segment.id, segment.speaker) for segment in assigned] == [
            ("segment-2", "S2"),
            ("segment-1", "S1"),
        ]

    def test_ignores_expired_turns_after_active_overlapping_turn(self) -> None:
        assigned = assign_speakers(
            [_segment("segment-1", 5.0, 6.0)],
            [
                SpeakerTurn(start_sec=0.0, end_sec=10.0, speaker="S1"),
                SpeakerTurn(start_sec=1.0, end_sec=2.0, speaker="S2"),
            ],
        )

        assert assigned[0].speaker == "S1"


def test_normalize_speaker_labels_orders_by_first_appearance() -> None:
    turns = normalize_speaker_labels([
        SpeakerTurn(start_sec=3.0, end_sec=4.0, speaker="17"),
        SpeakerTurn(start_sec=1.0, end_sec=2.0, speaker="42"),
        SpeakerTurn(start_sec=2.0, end_sec=3.0, speaker="17"),
    ])

    assert [(turn.start_sec, turn.speaker) for turn in turns] == [
        (1.0, "S1"),
        (2.0, "S2"),
        (3.0, "S2"),
    ]


class TestConfig:
    def test_default_cache_dir_uses_home_cache(self) -> None:
        cache_dir = default_cache_dir()

        assert cache_dir.name == "diarization"
        assert cache_dir.parent.name == "webinar-transcriber"

    def test_default_cache_dir_uses_env_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("WEBINAR_DIARIZATION_CACHE_DIR", str(tmp_path))

        assert default_cache_dir() == tmp_path

    def test_default_model_paths_use_configured_cache_dir(self, tmp_path: Path) -> None:
        paths = default_model_paths(tmp_path)

        assert (
            paths.segmentation_model
            == tmp_path / SEGMENTATION_MODEL.directory / SEGMENTATION_MODEL.file_name
        )
        assert paths.embedding_model == tmp_path / "nemo_en_titanet_small.onnx"


class TestSherpaOnnxDiarizer:
    def test_errors_when_sherpa_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sherpa_runtime, "_load_sherpa_onnx", lambda: None)

        with pytest.raises(DiarizationProcessingError, match="sherpa-onnx is unavailable"):
            sherpa_runtime.SherpaOnnxDiarizer(threads=1).prepare(speaker_count=2)

    def test_runs_auto_clustering_once_and_normalizes_labels(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_sherpa = FakeSherpaModule(
            results=[
                [
                    FakeSherpaItem(0.0, 1.0, 9),
                    FakeSherpaItem(1.0, 2.0, 8),
                    FakeSherpaItem(2.0, 2.0, 9),
                ]
            ]
        )
        monkeypatch.setattr(sherpa_runtime, "_load_sherpa_onnx", lambda: fake_sherpa)
        monkeypatch.setattr(
            sherpa_runtime,
            "ensure_default_models",
            lambda _cache_dir: sherpa_runtime.DiarizationModelPaths(
                segmentation_model=tmp_path / "seg.onnx", embedding_model=tmp_path / "emb.onnx"
            ),
        )
        progress: list[tuple[int, int]] = []

        diarizer = sherpa_runtime.SherpaOnnxDiarizer(cache_dir=tmp_path, threads=3)
        diarizer.prepare(speaker_count=None)

        turns = diarizer.diarize(
            np.zeros(16, dtype=np.float32),
            progress_callback=lambda processed, total: progress.append((processed, total)),
        )

        assert fake_sherpa.cluster_counts == [-1]
        assert fake_sherpa.thread_counts == [3, 3]
        assert turns == [
            SpeakerTurn(start_sec=0.0, end_sec=1.0, speaker="S1"),
            SpeakerTurn(start_sec=1.0, end_sec=2.0, speaker="S2"),
        ]
        assert progress == [(1, 2), (2, 2)]

    def test_uses_known_speaker_count_as_exact_cluster_count(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_sherpa = FakeSherpaModule(
            results=[[FakeSherpaItem(0.0, 1.0, 9), FakeSherpaItem(1.0, 2.0, 8)]]
        )
        monkeypatch.setattr(sherpa_runtime, "_load_sherpa_onnx", lambda: fake_sherpa)
        monkeypatch.setattr(
            sherpa_runtime,
            "ensure_default_models",
            lambda _cache_dir: sherpa_runtime.DiarizationModelPaths(
                segmentation_model=tmp_path / "seg.onnx", embedding_model=tmp_path / "emb.onnx"
            ),
        )

        diarizer = sherpa_runtime.SherpaOnnxDiarizer(cache_dir=tmp_path, threads=1)
        diarizer.prepare(speaker_count=2)

        turns = diarizer.diarize(np.zeros(16, dtype=np.float32))

        assert fake_sherpa.cluster_counts == [2]
        assert [turn.speaker for turn in turns] == ["S1", "S2"]

    def test_exposes_system_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sherpa_runtime.metadata, "version", lambda _name: "1.2.3")

        diarizer = sherpa_runtime.SherpaOnnxDiarizer(threads=1)

        assert diarizer.system_info == "sherpa-onnx 1.2.3"

    def test_prepare_rejects_invalid_config(self, tmp_path: Path) -> None:
        diarizer = sherpa_runtime.SherpaOnnxDiarizer(cache_dir=tmp_path, threads=1)

        with pytest.raises(DiarizationProcessingError, match="configuration is invalid"):
            diarizer._build_diarizer(  # noqa: SLF001
                FakeSherpaModule(valid=False),
                paths=sherpa_runtime.DiarizationModelPaths(
                    segmentation_model=tmp_path / "seg.onnx", embedding_model=tmp_path / "emb.onnx"
                ),
                num_clusters=-1,
            )

    def test_prepare_wraps_native_constructor_error(self, tmp_path: Path) -> None:
        class BrokenConstructorSherpa(FakeSherpaModule):
            def OfflineSpeakerDiarization(self, _config):  # noqa: N802
                raise RuntimeError("native constructor failure")

        diarizer = sherpa_runtime.SherpaOnnxDiarizer(cache_dir=tmp_path, threads=1)

        with pytest.raises(DiarizationProcessingError, match="native constructor failure"):
            diarizer._build_diarizer(  # noqa: SLF001
                BrokenConstructorSherpa(),
                paths=sherpa_runtime.DiarizationModelPaths(
                    segmentation_model=tmp_path / "seg.onnx", embedding_model=tmp_path / "emb.onnx"
                ),
                num_clusters=-1,
            )

    def test_diarize_raises_when_not_prepared(self) -> None:
        diarizer = sherpa_runtime.SherpaOnnxDiarizer(threads=1)

        with pytest.raises(DiarizationProcessingError, match="not prepared"):
            diarizer.diarize(np.zeros(16, dtype=np.float32))

    def test_run_diarization_wraps_runtime_error(self) -> None:
        diarizer = sherpa_runtime.SherpaOnnxDiarizer(threads=1)
        fake_sherpa = FakeSherpaModule(runtime_error=RuntimeError("native failure"))

        with pytest.raises(DiarizationProcessingError, match="native failure"):
            diarizer._run_diarization(  # noqa: SLF001
                fake_sherpa.OfflineSpeakerDiarization(None),
                np.zeros(16, dtype=np.float32),
                progress_callback=None,
            )


class TestModelDownload:
    def test_ensure_default_models_downloads_both_model_types(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        paths = sherpa_runtime.DiarizationModelPaths(
            segmentation_model=tmp_path / "seg.onnx", embedding_model=tmp_path / "emb.onnx"
        )
        calls: list[Path] = []
        monkeypatch.setattr(sherpa_runtime, "default_model_paths", lambda _cache_dir: paths)
        monkeypatch.setattr(
            sherpa_runtime, "_ensure_segmentation_model", lambda path: calls.append(Path(path))
        )
        monkeypatch.setattr(
            sherpa_runtime, "_ensure_file", lambda path, **_kwargs: calls.append(Path(path))
        )

        assert sherpa_runtime.ensure_default_models(tmp_path) == paths
        assert calls == [paths.segmentation_model, paths.embedding_model]

    def test_ensure_file_downloads_and_verifies(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        payload = b"model-bytes"
        assert FakeResponse(payload).read() == payload
        monkeypatch.setattr(
            sherpa_runtime.urllib.request, "urlopen", lambda _url: FakeResponse(payload)
        )

        output_path = tmp_path / "model.onnx"
        sherpa_runtime._ensure_file(  # noqa: SLF001
            output_path, url="https://example.test/model", expected_sha256=_sha256(payload)
        )

        assert output_path.read_bytes() == payload
        sherpa_runtime._ensure_file(  # noqa: SLF001
            output_path, url="https://example.test/model", expected_sha256=_sha256(payload)
        )

    def test_ensure_file_reports_download_and_hash_failures(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            sherpa_runtime.urllib.request, "urlopen", Mock(side_effect=OSError("offline"))
        )
        with pytest.raises(DiarizationProcessingError, match="Failed to download"):
            sherpa_runtime._ensure_file(  # noqa: SLF001
                tmp_path / "offline.onnx",
                url="https://example.test/model",
                expected_sha256="missing",
            )

        monkeypatch.setattr(
            sherpa_runtime.urllib.request, "urlopen", lambda _url: FakeResponse(b"wrong")
        )
        with pytest.raises(DiarizationProcessingError, match="failed verification"):
            sherpa_runtime._ensure_file(  # noqa: SLF001
                tmp_path / "bad.onnx", url="https://example.test/model", expected_sha256="missing"
            )

    def test_ensure_segmentation_model_extracts_archive(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        model_bytes = b"segmentation-model"
        archive_path = tmp_path / f"{SEGMENTATION_MODEL.directory}.tar.bz2"
        model_path = tmp_path / SEGMENTATION_MODEL.directory / SEGMENTATION_MODEL.file_name
        with tarfile.open(archive_path, "w:bz2") as archive:
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(model_bytes)
            archive.add(
                model_path, arcname=f"{SEGMENTATION_MODEL.directory}/{SEGMENTATION_MODEL.file_name}"
            )
        model_path.unlink()
        monkeypatch.setattr(
            sherpa_runtime,
            "SEGMENTATION_MODEL",
            replace(SEGMENTATION_MODEL, model_sha256=_sha256(model_bytes)),
        )
        monkeypatch.setattr(
            sherpa_runtime,
            "_ensure_file",
            lambda path, **_kwargs: Path(path).write_bytes(archive_path.read_bytes()),
        )

        sherpa_runtime._ensure_segmentation_model(model_path)  # noqa: SLF001

        assert model_path.read_bytes() == model_bytes
        sherpa_runtime._ensure_segmentation_model(model_path)  # noqa: SLF001

    def test_ensure_segmentation_model_reports_bad_extract(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        archive_path = tmp_path / f"{SEGMENTATION_MODEL.directory}.tar.bz2"
        model_path = tmp_path / SEGMENTATION_MODEL.directory / SEGMENTATION_MODEL.file_name
        with tarfile.open(archive_path, "w:bz2") as archive:
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"segmentation-model")
            archive.add(
                model_path, arcname=f"{SEGMENTATION_MODEL.directory}/{SEGMENTATION_MODEL.file_name}"
            )
        model_path.unlink()
        monkeypatch.setattr(
            sherpa_runtime, "SEGMENTATION_MODEL", replace(SEGMENTATION_MODEL, model_sha256="wrong")
        )
        monkeypatch.setattr(
            sherpa_runtime,
            "_ensure_file",
            lambda path, **_kwargs: Path(path).write_bytes(archive_path.read_bytes()),
        )

        with pytest.raises(DiarizationProcessingError, match="failed verification"):
            sherpa_runtime._ensure_segmentation_model(model_path)  # noqa: SLF001

    def test_load_sherpa_onnx_handles_missing_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sherpa_runtime.importlib, "import_module", Mock(side_effect=ImportError)
        )

        assert sherpa_runtime._load_sherpa_onnx() is None  # noqa: SLF001

    def test_load_sherpa_onnx_returns_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_module = object()
        monkeypatch.setattr(sherpa_runtime.importlib, "import_module", lambda _name: fake_module)

        assert sherpa_runtime._load_sherpa_onnx() is fake_module  # noqa: SLF001
