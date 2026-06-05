"""sherpa-onnx speaker-diarization backend."""

from __future__ import annotations

import hashlib
import importlib
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from webinar_transcriber._env import diarization_cache_dir
from webinar_transcriber.models import SpeakerTurn

from .contracts import DiarizationProcessingError

CLUSTER_THRESHOLD = 0.5
MIN_DURATION_ON_SEC = 0.3
MIN_DURATION_OFF_SEC = 0.5
DIARIZATION_MODEL = "pyannote-segmentation-3.0-fp32+nemo-titanet-small"


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
    from typing import Protocol

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


class SherpaOnnxDiarizer:
    """Local speaker diarizer backed by sherpa-onnx."""

    def __init__(self, *, threads: int, cache_dir: Path | None = None) -> None:
        """Initialize a lazy sherpa-onnx diarizer."""
        self._cache_dir = cache_dir or default_cache_dir()
        self._threads = threads
        self._prepared_diarizer: _NativeDiarizer | None = None

    @property
    def system_info(self) -> str | None:
        """Return the sherpa-onnx version when installed."""
        try:
            return f"sherpa-onnx {metadata.version('sherpa-onnx')}"
        except metadata.PackageNotFoundError:  # pragma: no cover - boundary dependency state
            return None

    def diarize(
        self, samples: np.ndarray, *, progress_callback: Callable[[int, int], None] | None = None
    ) -> list[SpeakerTurn]:
        """Return speaker turns for normalized audio samples."""
        if self._prepared_diarizer is None:
            raise DiarizationProcessingError("Diarizer not prepared; call prepare() first.")
        diarizer = self._prepared_diarizer

        turns = self._run_diarization(diarizer, samples, progress_callback=progress_callback)
        return normalize_speaker_labels(turns)

    def prepare(self, *, speaker_count: int | None) -> None:
        """Load diarization models and construct the native diarizer."""
        sherpa_onnx = _load_sherpa_onnx()
        if sherpa_onnx is None:
            raise DiarizationProcessingError("sherpa-onnx is unavailable for speaker diarization.")

        paths = ensure_default_models(self._cache_dir)
        num_clusters = speaker_count or -1
        self._prepared_diarizer = self._build_diarizer(
            sherpa_onnx, paths=paths, num_clusters=num_clusters
        )

    def _build_diarizer(
        self, sherpa_onnx: ModuleType, *, paths: DiarizationModelPaths, num_clusters: int
    ) -> _NativeDiarizer:
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(paths.segmentation_model)
                ),
                num_threads=self._threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(paths.embedding_model), num_threads=self._threads
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_clusters, threshold=CLUSTER_THRESHOLD
            ),
            min_duration_on=MIN_DURATION_ON_SEC,
            min_duration_off=MIN_DURATION_OFF_SEC,
        )
        if not config.validate():
            raise DiarizationProcessingError("Speaker diarization model configuration is invalid.")

        try:
            return sherpa_onnx.OfflineSpeakerDiarization(config)
        except RuntimeError as error:
            raise DiarizationProcessingError(str(error)) from error

    def _run_diarization(
        self,
        diarizer: _NativeDiarizer,
        samples: np.ndarray,
        *,
        progress_callback: Callable[[int, int], None] | None,
    ) -> list[SpeakerTurn]:
        try:
            result = diarizer.process(samples, callback=_sherpa_progress(progress_callback))
        except RuntimeError as error:
            raise DiarizationProcessingError(str(error)) from error

        return [
            SpeakerTurn(
                start_sec=float(item.start), end_sec=float(item.end), speaker=str(item.speaker)
            )
            for item in result.sort_by_start_time()
            if float(item.end) > float(item.start)
        ]


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


def normalize_speaker_labels(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Return turns with stable S1..SN labels ordered by first appearance."""
    labels: dict[str, str] = {}
    normalized: list[SpeakerTurn] = []
    for turn in sorted(turns, key=lambda item: (item.start_sec, item.end_sec)):
        if turn.speaker not in labels:
            labels[turn.speaker] = f"S{len(labels) + 1}"
        normalized.append(replace(turn, speaker=labels[turn.speaker]))
    return normalized


def _sherpa_progress(
    progress_callback: Callable[[int, int], None] | None,
) -> Callable[[int, int], int] | None:
    if progress_callback is None:
        return None

    def callback(processed_chunks: int, total_chunks: int) -> int:
        progress_callback(processed_chunks, total_chunks)
        return 0

    return callback


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
        raise DiarizationProcessingError(
            f"Downloaded diarization model failed verification: {model_path}"
        )


def _ensure_file(path: Path, *, url: str, expected_sha256: str) -> None:
    if _verified(path, expected_sha256):
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            urllib.request.urlopen(url) as response,  # noqa: S310
            NamedTemporaryFile(dir=path.parent, delete=False) as temp_file,
        ):
            temp_path = Path(temp_file.name)
            while chunk := response.read(1024 * 1024):
                temp_file.write(chunk)
    except (OSError, urllib.error.URLError) as error:
        raise DiarizationProcessingError(f"Failed to download diarization model: {url}") from error

    if not _verified(temp_path, expected_sha256):
        temp_path.unlink(missing_ok=True)
        raise DiarizationProcessingError(f"Downloaded diarization model failed verification: {url}")
    temp_path.replace(path)


def _verified(path: Path, expected_sha256: str) -> bool:
    if not path.exists():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        while chunk := model_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def _load_sherpa_onnx() -> ModuleType | None:
    try:
        return importlib.import_module("sherpa_onnx")
    except ImportError:  # pragma: no cover - optional wheel/import boundary
        return None
