"""Tests for local speaker-diarization helpers."""

from __future__ import annotations

import hashlib
import tarfile
from dataclasses import replace
from pathlib import Path
from queue import Empty
from types import ModuleType, SimpleNamespace
from typing import Self
from unittest.mock import Mock

import numpy as np
import pytest

import webinar_transcriber.diarization.sherpa_diarizer as sherpa_runtime
from webinar_transcriber.diarization import DiarizationProcessingError, assign_speakers
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


# get() script sentinels for the queue double.
_TIMEOUT = object()  # simulate a poll timeout (raises Empty)
_INTERRUPT = object()  # simulate Ctrl-C during the blocking get


class _FakeQueue:
    """Queue double: the child records puts; the parent replays scripted get()/get_nowait()."""

    def __init__(self, get_script: list | None = None, nowait_script: list | None = None) -> None:
        self._get = list(get_script or [])
        self._nowait = list(nowait_script or [])
        self.puts: list[object] = []

    def put(self, item: object) -> None:
        self.puts.append(item)

    def get(self, *, timeout: float | None = None) -> object:
        del timeout
        item = self._get.pop(0)
        if item is _TIMEOUT:
            raise Empty
        if item is _INTERRUPT:
            raise KeyboardInterrupt
        return item

    def get_nowait(self) -> object:
        if not self._nowait:
            raise Empty
        return self._nowait.pop(0)


class _FakeProcess:
    def __init__(self, alive: list[bool] | None = None) -> None:
        self._alive = list(alive or [])
        self.started = False
        self.terminated = False
        self.joins = 0

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self._alive.pop(0)

    def terminate(self) -> None:
        self.terminated = True

    def join(self) -> None:
        self.joins += 1


class _FakeContext:
    def __init__(self, queue: _FakeQueue, process: _FakeProcess) -> None:
        self._queue = queue
        self._process = process
        self.process_kwargs: dict | None = None

    def Queue(self) -> _FakeQueue:  # noqa: N802 - mirrors multiprocessing context API
        return self._queue

    def Process(self, *, target, args, kwargs) -> _FakeProcess:  # noqa: N802 - mirrors the API
        del target, args
        self.process_kwargs = kwargs
        return self._process


def _subprocess_kwargs(tmp_path: Path, *, num_clusters: int = -1, threads: int = 3) -> dict:
    return {
        "segmentation_model": tmp_path / "seg.onnx",
        "embedding_model": tmp_path / "emb.onnx",
        "num_clusters": num_clusters,
        "threads": threads,
        "wav_path": tmp_path / "audio.wav",
    }


class TestDiarizationSubprocessTarget:
    """The child entry point, exercised in-process (a real spawn would escape coverage)."""

    def test_streams_progress_then_raw_turns(
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
        monkeypatch.setattr(sherpa_runtime, "load_sherpa_onnx", lambda: fake_sherpa)
        monkeypatch.setattr(
            sherpa_runtime, "load_normalized_audio", lambda _path: np.zeros(16, dtype=np.float32)
        )
        queue = _FakeQueue()

        sherpa_runtime._run_diarization_subprocess(queue, **_subprocess_kwargs(tmp_path))  # noqa: SLF001

        assert fake_sherpa.cluster_counts == [-1]
        assert fake_sherpa.thread_counts == [3, 3]
        assert queue.puts == [
            ("progress", 1, 2),
            ("progress", 2, 2),
            (
                "done",
                [
                    SpeakerTurn(start_sec=0.0, end_sec=1.0, speaker="9"),
                    SpeakerTurn(start_sec=1.0, end_sec=2.0, speaker="8"),
                ],
            ),
        ]

    def test_passes_known_speaker_count_as_cluster_count(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_sherpa = FakeSherpaModule(results=[[FakeSherpaItem(0.0, 1.0, 9)]])
        monkeypatch.setattr(sherpa_runtime, "load_sherpa_onnx", lambda: fake_sherpa)
        monkeypatch.setattr(
            sherpa_runtime, "load_normalized_audio", lambda _path: np.zeros(16, dtype=np.float32)
        )
        queue = _FakeQueue()

        sherpa_runtime._run_diarization_subprocess(  # noqa: SLF001
            queue, **_subprocess_kwargs(tmp_path, num_clusters=2)
        )

        assert fake_sherpa.cluster_counts == [2]
        assert queue.puts[-1] == ("done", [SpeakerTurn(start_sec=0.0, end_sec=1.0, speaker="9")])

    def test_reports_missing_sherpa_as_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(sherpa_runtime, "load_sherpa_onnx", lambda: None)
        queue = _FakeQueue()

        sherpa_runtime._run_diarization_subprocess(queue, **_subprocess_kwargs(tmp_path))  # noqa: SLF001

        assert queue.puts == [("error", "sherpa-onnx is unavailable for speaker diarization.")]

    def test_reports_invalid_config_as_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            sherpa_runtime, "load_sherpa_onnx", lambda: FakeSherpaModule(valid=False)
        )
        monkeypatch.setattr(
            sherpa_runtime, "load_normalized_audio", lambda _path: np.zeros(16, dtype=np.float32)
        )
        queue = _FakeQueue()

        sherpa_runtime._run_diarization_subprocess(queue, **_subprocess_kwargs(tmp_path))  # noqa: SLF001

        assert queue.puts == [("error", "Speaker diarization model configuration is invalid.")]

    def test_reports_native_constructor_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        class BrokenConstructorSherpa(FakeSherpaModule):
            def OfflineSpeakerDiarization(self, _config):  # noqa: N802
                raise RuntimeError("native constructor failure")

        monkeypatch.setattr(sherpa_runtime, "load_sherpa_onnx", BrokenConstructorSherpa)
        monkeypatch.setattr(
            sherpa_runtime, "load_normalized_audio", lambda _path: np.zeros(16, dtype=np.float32)
        )
        queue = _FakeQueue()

        sherpa_runtime._run_diarization_subprocess(queue, **_subprocess_kwargs(tmp_path))  # noqa: SLF001

        assert queue.puts == [("error", "native constructor failure")]

    def test_reports_native_process_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            sherpa_runtime,
            "load_sherpa_onnx",
            lambda: FakeSherpaModule(runtime_error=RuntimeError("native failure")),
        )
        monkeypatch.setattr(
            sherpa_runtime, "load_normalized_audio", lambda _path: np.zeros(16, dtype=np.float32)
        )
        queue = _FakeQueue()

        sherpa_runtime._run_diarization_subprocess(queue, **_subprocess_kwargs(tmp_path))  # noqa: SLF001

        assert queue.puts == [("error", "native failure")]


class TestSherpaOnnxDiarizer:
    def test_errors_when_sherpa_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sherpa_runtime, "load_sherpa_onnx", lambda: None)

        with pytest.raises(DiarizationProcessingError, match="sherpa-onnx is unavailable"):
            sherpa_runtime.SherpaOnnxDiarizer(threads=1).prepare(speaker_count=2)

    def test_exposes_system_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sherpa_runtime.metadata, "version", lambda _name: "1.2.3")

        diarizer = sherpa_runtime.SherpaOnnxDiarizer(threads=1)

        assert diarizer.system_info == "sherpa-onnx 1.2.3"

    def test_diarize_raises_when_not_prepared(self) -> None:
        diarizer = sherpa_runtime.SherpaOnnxDiarizer(threads=1)

        with pytest.raises(DiarizationProcessingError, match="not prepared"):
            diarizer.diarize(Path("unused.wav"))

    def _prepared_diarizer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        queue: _FakeQueue,
        process: _FakeProcess,
        speaker_count: int | None = None,
    ) -> tuple[sherpa_runtime.SherpaOnnxDiarizer, _FakeContext]:
        monkeypatch.setattr(sherpa_runtime, "load_sherpa_onnx", FakeSherpaModule)
        monkeypatch.setattr(
            sherpa_runtime,
            "ensure_default_models",
            lambda _cache_dir: sherpa_runtime.DiarizationModelPaths(
                segmentation_model=tmp_path / "seg.onnx", embedding_model=tmp_path / "emb.onnx"
            ),
        )
        context = _FakeContext(queue, process)
        monkeypatch.setattr(sherpa_runtime.multiprocessing, "get_context", lambda _method: context)
        diarizer = sherpa_runtime.SherpaOnnxDiarizer(cache_dir=tmp_path, threads=1)
        diarizer.prepare(speaker_count=speaker_count)
        return diarizer, context

    def test_diarize_streams_progress_and_normalizes_labels(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        queue = _FakeQueue(
            get_script=[
                ("progress", 1, 2),
                ("done", [SpeakerTurn(start_sec=0.0, end_sec=1.0, speaker="7")]),
            ]
        )
        process = _FakeProcess()
        diarizer, context = self._prepared_diarizer(
            monkeypatch, tmp_path, queue=queue, process=process
        )
        progress: list[tuple[int, int]] = []

        turns = diarizer.diarize(
            tmp_path / "audio.wav",
            progress_callback=lambda processed, total: progress.append((processed, total)),
        )

        assert progress == [(1, 2)]
        assert turns == [SpeakerTurn(start_sec=0.0, end_sec=1.0, speaker="S1")]
        assert process.started
        assert process.joins == 1
        assert not process.terminated
        assert context.process_kwargs is not None
        assert context.process_kwargs["wav_path"] == tmp_path / "audio.wav"
        assert context.process_kwargs["num_clusters"] == -1

    def test_diarize_raises_and_terminates_on_subprocess_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        process = _FakeProcess()
        diarizer, _ = self._prepared_diarizer(
            monkeypatch,
            tmp_path,
            queue=_FakeQueue(get_script=[("error", "subprocess boom")]),
            process=process,
        )

        with pytest.raises(DiarizationProcessingError, match="subprocess boom"):
            diarizer.diarize(tmp_path / "audio.wav")

        assert process.terminated
        assert process.joins == 1

    def test_diarize_raises_when_child_exits_without_result(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        process = _FakeProcess(alive=[False])
        diarizer, _ = self._prepared_diarizer(
            monkeypatch, tmp_path, queue=_FakeQueue(get_script=[_TIMEOUT]), process=process
        )

        with pytest.raises(DiarizationProcessingError, match="exited without returning a result"):
            diarizer.diarize(tmp_path / "audio.wav")

        assert process.terminated

    def test_diarize_keeps_polling_while_child_is_alive(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        queue = _FakeQueue(
            get_script=[_TIMEOUT, ("done", [SpeakerTurn(start_sec=0.0, end_sec=1.0, speaker="7")])]
        )
        diarizer, _ = self._prepared_diarizer(
            monkeypatch, tmp_path, queue=queue, process=_FakeProcess(alive=[True])
        )

        turns = diarizer.diarize(tmp_path / "audio.wav")

        assert turns == [SpeakerTurn(start_sec=0.0, end_sec=1.0, speaker="S1")]

    def test_diarize_recovers_final_message_after_child_exit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        queue = _FakeQueue(
            get_script=[_TIMEOUT],
            nowait_script=[("done", [SpeakerTurn(start_sec=0.0, end_sec=1.0, speaker="7")])],
        )
        diarizer, _ = self._prepared_diarizer(
            monkeypatch, tmp_path, queue=queue, process=_FakeProcess(alive=[False])
        )

        turns = diarizer.diarize(tmp_path / "audio.wav")

        assert turns == [SpeakerTurn(start_sec=0.0, end_sec=1.0, speaker="S1")]

    def test_diarize_terminates_child_on_interrupt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        process = _FakeProcess()
        diarizer, _ = self._prepared_diarizer(
            monkeypatch, tmp_path, queue=_FakeQueue(get_script=[_INTERRUPT]), process=process
        )

        with pytest.raises(KeyboardInterrupt):
            diarizer.diarize(tmp_path / "audio.wav")

        assert process.terminated
        assert process.joins == 1


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
