"""sherpa-onnx speaker-diarization backend."""

from __future__ import annotations

import hashlib
import multiprocessing
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from webinar_transcriber._env import diarization_cache_dir, load_sherpa_onnx
from webinar_transcriber.models import SpeakerTurn
from webinar_transcriber.normalized_audio import load_normalized_audio

CLUSTER_THRESHOLD = 1.2
MIN_DURATION_ON_SEC = 0.5
# A speaker whose longest turn never reaches an utterance is a boundary artifact, not a voice
MIN_SPEAKER_TURN_SEC = 4 * MIN_DURATION_ON_SEC
MIN_DURATION_OFF_SEC = 1.5
DIARIZATION_MODEL = "pyannote-segmentation-3.0-fp32+nemo-titanet-small"
# Poll often enough to detect a child that exits without a terminal message
DIARIZATION_POLL_SEC = 0.5
MODEL_DOWNLOAD_TIMEOUT_SEC = 60


class DiarizationProcessingError(RuntimeError):
    """Raised when local speaker diarization cannot complete."""


class DiarizationConfigurationError(DiarizationProcessingError):
    """Raised when the diarization models or runtime cannot be set up.

    Separate from a per-run diarization failure because it depends on the host and cached models
    rather than on the audio, so every input in a batch would hit it identically.
    """


@dataclass(frozen=True)
class _SegmentationModel:
    directory: str
    file_name: str
    archive_url: str
    archive_sha256: str
    model_sha256: str


@dataclass(frozen=True)
class _EmbeddingModel:
    file_name: str
    url: str
    sha256: str


SEGMENTATION_MODEL = _SegmentationModel(
    directory="sherpa-onnx-pyannote-segmentation-3-0",
    file_name="model.onnx",
    archive_url="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    archive_sha256="24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
    model_sha256="220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
)
EMBEDDING_MODEL = _EmbeddingModel(
    file_name="nemo_en_titanet_small.onnx",
    url="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx",
    sha256="ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
)


@dataclass(frozen=True)
class DiarizationModelPaths:
    """Resolved local model paths for sherpa-onnx diarization."""

    segmentation_model: Path
    embedding_model: Path


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any, Protocol

    import numpy as np

    class _DiarizationItem(Protocol):
        start: float
        end: float
        speaker: object

    class _DiarizationResult(Protocol):
        def sort_by_start_time(self) -> list[_DiarizationItem]: ...

    class _NativeDiarizer(Protocol):
        def process(
            self, samples: np.ndarray, *, callback: Callable[[int, int], int] | None = None
        ) -> _DiarizationResult: ...

    class _MessageQueue(Protocol):
        """Queue operations used to exchange messages between the parent and child processes."""

        def put(self, item: object, /) -> None: ...
        def get(self, *, timeout: float | None = ...) -> Any: ...  # noqa: ANN401 - heterogeneous (kind, *payload) tuple
        def get_nowait(self) -> Any: ...  # noqa: ANN401 - heterogeneous (kind, *payload) tuple

    class _DiarizationProcess(Protocol):
        """Process state used to detect a child that exits without a terminal message."""

        def is_alive(self) -> bool: ...


class SherpaOnnxDiarizer:
    """Local speaker diarizer backed by sherpa-onnx."""

    def __init__(self, *, threads: int, cache_dir: Path | None = None) -> None:
        """Initialize a lazy sherpa-onnx diarizer."""
        self._cache_dir = cache_dir or default_cache_dir()
        self._threads = threads
        self._model_paths: DiarizationModelPaths | None = None
        self._speaker_count: int | None = None

    @property
    def system_info(self) -> str | None:
        """Return the sherpa-onnx version when installed."""
        try:
            return f"sherpa-onnx {metadata.version('sherpa-onnx')}"
        except metadata.PackageNotFoundError:  # pragma: no cover - boundary dependency state
            return None

    def prepare(self, *, speaker_count: int | None) -> None:
        """Resolve diarization models; the native diarizer is built in the subprocess.

        A known speaker count is applied after clustering rather than handed to sherpa-onnx, which
        ignores ``threshold`` whenever ``num_clusters`` is set. Cutting the dendrogram at the human
        speaker count discards the tuned ``CLUSTER_THRESHOLD`` and strands non-speech clusters,
        which merges distinct speakers instead of dropping the noise.
        """
        if load_sherpa_onnx() is None:
            raise DiarizationConfigurationError(
                "sherpa-onnx is unavailable for speaker diarization."
            )
        self._model_paths = ensure_default_models(self._cache_dir)
        self._speaker_count = speaker_count

    def diarize(
        self, wav_path: Path, *, progress_callback: Callable[[int, int], None] | None = None
    ) -> list[SpeakerTurn]:
        """Run diarization in a child process and return normalized speaker turns.

        sherpa-onnx ``process()`` holds the GIL for its whole run, so running it here would freeze
        the progress display and block Ctrl-C. The child does the work and streams progress back
        over a queue; this main thread stays responsive, so the bar animates and a Ctrl-C
        terminates the child.
        """
        if self._model_paths is None:
            raise DiarizationConfigurationError("Diarizer not prepared; call prepare() first.")

        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        process = context.Process(
            target=_run_diarization_subprocess,
            args=(queue,),
            kwargs={
                "segmentation_model": self._model_paths.segmentation_model,
                "embedding_model": self._model_paths.embedding_model,
                "threads": self._threads,
                "wav_path": wav_path,
            },
        )
        process.start()
        try:
            turns = _drain_diarization(queue, process, progress_callback=progress_callback)
        except BaseException:
            process.terminate()
            process.join()
            raise
        process.join()
        turns = drop_spurious_speakers(turns)
        if self._speaker_count is not None:
            turns = reconcile_speaker_count(turns, self._speaker_count)
        return normalize_speaker_labels(turns)


def ensure_default_models(cache_dir: Path | None = None) -> DiarizationModelPaths:
    """Download and verify default diarization models when missing."""
    paths = default_model_paths(cache_dir)
    _ensure_segmentation_model(paths.segmentation_model)
    _ensure_file(
        paths.embedding_model, url=EMBEDDING_MODEL.url, expected_sha256=EMBEDDING_MODEL.sha256
    )
    return paths


def default_cache_dir() -> Path:
    """Return the speaker-diarization model cache directory."""
    return diarization_cache_dir() or Path.home() / ".cache" / "webinar-transcriber" / "diarization"


def default_model_paths(cache_dir: Path | None = None) -> DiarizationModelPaths:
    """Return expected local model paths under the configured cache directory."""
    root = cache_dir or default_cache_dir()
    return DiarizationModelPaths(
        segmentation_model=root / SEGMENTATION_MODEL.directory / SEGMENTATION_MODEL.file_name,
        embedding_model=root / EMBEDDING_MODEL.file_name,
    )


def reconcile_speaker_count(turns: list[SpeakerTurn], speaker_count: int) -> list[SpeakerTurn]:
    """Fold the shortest-speaking speakers into their neighbours until ``speaker_count`` remain.

    Clustering runs at the tuned threshold and can strand short non-speech clusters, so a recording
    with two speakers may come back with three. Reassigning those turns preserves the threshold's
    separation of the real speakers, which re-clustering at a fixed count does not. Fewer speakers
    than asked for are left alone; a cluster that was never found cannot be recovered here.
    """
    speaking_time: dict[str, float] = {}
    for turn in turns:
        speaking_time[turn.speaker] = speaking_time.get(turn.speaker, 0.0) + turn.duration_sec
    if len(speaking_time) <= speaker_count:
        return turns

    ranked = sorted(speaking_time, key=lambda speaker: (-speaking_time[speaker], speaker))
    return _fold_into(turns, set(ranked[:speaker_count]))


def drop_spurious_speakers(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Fold speakers who never hold the floor for a whole utterance into their neighbours.

    Clustering at the tuned threshold strands momentary boundary artifacts as their own speaker: on
    a 2.2h two-presenter recording it reported a third speaker holding 3.2s across turns of 0.9s,
    0.6s, and 1.7s. Judging on the longest turn rather than on total speech keeps a real participant
    who only asks one question, whose single utterance runs far longer than any artifact.
    """
    longest_turn: dict[str, float] = {}
    for turn in turns:
        longest_turn[turn.speaker] = max(longest_turn.get(turn.speaker, 0.0), turn.duration_sec)

    retained = {
        speaker for speaker, longest in longest_turn.items() if longest >= MIN_SPEAKER_TURN_SEC
    }
    if not retained or len(retained) == len(longest_turn):
        return turns
    return _fold_into(turns, retained)


def _fold_into(turns: list[SpeakerTurn], retained: set[str]) -> list[SpeakerTurn]:
    anchors = [turn for turn in turns if turn.speaker in retained]
    return [
        turn if turn.speaker in retained else replace(turn, speaker=_nearest_speaker(turn, anchors))
        for turn in turns
    ]


def _nearest_speaker(turn: SpeakerTurn, anchors: list[SpeakerTurn]) -> str:
    """Return the speaker of the closest anchor turn, preferring the earlier one on a tie."""
    return min(anchors, key=lambda anchor: (_gap_sec(turn, anchor), anchor.start_sec)).speaker


def _gap_sec(turn: SpeakerTurn, other: SpeakerTurn) -> float:
    return max(0.0, other.start_sec - turn.end_sec, turn.start_sec - other.end_sec)


def normalize_speaker_labels(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Return turns with stable S1..SN labels ordered by first appearance."""
    labels: dict[str, str] = {}
    normalized: list[SpeakerTurn] = []
    for turn in sorted(turns, key=lambda item: (item.start_sec, item.end_sec)):
        if turn.speaker not in labels:
            labels[turn.speaker] = f"S{len(labels) + 1}"
        normalized.append(replace(turn, speaker=labels[turn.speaker]))
    return normalized


def _drain_diarization(
    queue: _MessageQueue,
    process: _DiarizationProcess,
    *,
    progress_callback: Callable[[int, int], None] | None,
) -> list[SpeakerTurn]:
    """Consume the child's messages, returning its turns or raising on failure.

    Polls with a timeout so a child that dies without a terminal message (a native segfault/SIGKILL
    or a bootstrap failure) raises instead of blocking forever; a last non-blocking read recovers a
    final message that raced the child's exit.
    """
    while True:
        try:
            message = queue.get(timeout=DIARIZATION_POLL_SEC)
        except Empty:
            if process.is_alive():
                continue
            try:
                message = queue.get_nowait()
            except Empty:
                raise DiarizationProcessingError(
                    "Diarization subprocess exited without returning a result."
                ) from None
        kind, *payload = message
        if kind == "progress":
            if progress_callback is not None:
                progress_callback(payload[0], payload[1])
        elif kind == "done":
            return payload[0]
        elif kind == "setup_error":
            raise DiarizationConfigurationError(payload[0])
        else:
            raise DiarizationProcessingError(payload[0])


def _run_diarization_subprocess(
    queue: _MessageQueue,
    *,
    segmentation_model: Path,
    embedding_model: Path,
    threads: int,
    wav_path: Path,
) -> None:
    """Child entry point: build the native diarizer, run it, and stream results over the queue.

    Runs in a spawned process, so it re-imports sherpa-onnx and reloads the audio from disk rather
    than inheriting parent state. Any failure is reported back as an ``error`` message so the parent
    never blocks waiting for a result.
    """
    try:
        sherpa_onnx = load_sherpa_onnx()
        if sherpa_onnx is None:
            raise DiarizationConfigurationError(
                "sherpa-onnx is unavailable for speaker diarization."
            )
        diarizer = _build_native_diarizer(
            sherpa_onnx,
            paths=DiarizationModelPaths(
                segmentation_model=segmentation_model, embedding_model=embedding_model
            ),
            threads=threads,
        )
        samples = load_normalized_audio(wav_path)

        def callback(processed: int, total: int) -> int:
            queue.put(("progress", processed, total))
            return 0

        result = diarizer.process(samples, callback=callback)
        queue.put(("done", _turns_from_result(result)))
    except DiarizationConfigurationError as ex:
        # Exception types do not survive the queue, so tag setup failures for the parent to re-raise
        queue.put(("setup_error", str(ex)))
    except Exception as ex:  # noqa: BLE001 - process boundary: report any failure to the parent
        queue.put(("error", str(ex)))


def _build_native_diarizer(
    sherpa_onnx: ModuleType, *, paths: DiarizationModelPaths, threads: int
) -> _NativeDiarizer:
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(paths.segmentation_model)
            ),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(paths.embedding_model), num_threads=threads
        ),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=CLUSTER_THRESHOLD),
        min_duration_on=MIN_DURATION_ON_SEC,
        min_duration_off=MIN_DURATION_OFF_SEC,
    )
    if not config.validate():
        raise DiarizationConfigurationError("Speaker diarization model configuration is invalid.")

    try:
        return sherpa_onnx.OfflineSpeakerDiarization(config)
    except RuntimeError as ex:
        raise DiarizationConfigurationError(str(ex)) from ex


def _turns_from_result(result: _DiarizationResult) -> list[SpeakerTurn]:
    return [
        SpeakerTurn(start_sec=float(item.start), end_sec=float(item.end), speaker=str(item.speaker))
        for item in result.sort_by_start_time()
        if float(item.end) > float(item.start)
    ]


def _ensure_segmentation_model(model_path: Path) -> None:
    if _verified(model_path, SEGMENTATION_MODEL.model_sha256):
        return

    archive_path = model_path.parent.parent / f"{SEGMENTATION_MODEL.directory}.tar.bz2"
    _ensure_file(
        archive_path,
        url=SEGMENTATION_MODEL.archive_url,
        expected_sha256=SEGMENTATION_MODEL.archive_sha256,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:bz2") as archive:
        member = archive.getmember(f"{SEGMENTATION_MODEL.directory}/{SEGMENTATION_MODEL.file_name}")
        archive.extract(member, path=model_path.parent.parent, filter="data")

    if not _verified(model_path, SEGMENTATION_MODEL.model_sha256):
        raise DiarizationConfigurationError(
            f"Downloaded diarization model failed verification: {model_path}"
        )


def _ensure_file(path: Path, *, url: str, expected_sha256: str) -> None:
    if _verified(path, expected_sha256):
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp_dir:
        temp_path = Path(temp_dir) / path.name
        try:
            with (
                # The URL is a hardcoded HTTPS GitHub release asset, not user input
                urllib.request.urlopen(  # noqa: S310
                    url, timeout=MODEL_DOWNLOAD_TIMEOUT_SEC
                ) as response,
                temp_path.open("wb") as temp_file,
            ):
                while chunk := response.read(1024 * 1024):
                    temp_file.write(chunk)
        except (OSError, urllib.error.URLError) as ex:
            raise DiarizationConfigurationError(
                f"Failed to download diarization model: {url}"
            ) from ex

        if not _verified(temp_path, expected_sha256):
            raise DiarizationConfigurationError(
                f"Downloaded diarization model failed verification: {url}"
            )
        temp_path.replace(path)


def _verified(path: Path, expected_sha256: str) -> bool:
    if not path.exists():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        while chunk := model_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256
