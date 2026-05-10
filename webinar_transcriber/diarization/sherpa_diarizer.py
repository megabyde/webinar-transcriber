"""sherpa-onnx speaker-diarization backend."""

from __future__ import annotations

import hashlib
import importlib
import tarfile
import urllib.error
import urllib.request
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from webinar_transcriber.models import SpeakerTurn
from webinar_transcriber.normalized_audio import NORMALIZED_SAMPLE_RATE

from .config import (
    DEFAULT_CLUSTER_THRESHOLD,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_MIN_DURATION_OFF_SEC,
    DEFAULT_MIN_DURATION_ON_SEC,
    EMBEDDING_MODEL_SHA256,
    EMBEDDING_MODEL_URL,
    SEGMENTATION_ARCHIVE_SHA256,
    SEGMENTATION_ARCHIVE_URL,
    SEGMENTATION_MODEL_DIR,
    SEGMENTATION_MODEL_FILE,
    SEGMENTATION_MODEL_SHA256,
    DiarizationModelPaths,
    default_cache_dir,
    default_model_paths,
)
from .contracts import DiarizationProcessingError

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import numpy as np


class SherpaOnnxDiarizer:
    """Local speaker diarizer backed by sherpa-onnx."""

    def __init__(self, *, cache_dir: Path | None = None) -> None:
        """Initialize a lazy sherpa-onnx diarizer."""
        self._cache_dir = cache_dir or default_cache_dir()

    @property
    def model_name(self) -> str:
        """Return a human-facing model label."""
        return DEFAULT_DIARIZATION_MODEL

    @property
    def system_info(self) -> str | None:
        """Return the sherpa-onnx version when installed."""
        try:
            return f"sherpa-onnx {metadata.version('sherpa-onnx')}"
        except metadata.PackageNotFoundError:  # pragma: no cover - boundary dependency state
            return None

    def diarize(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        max_speakers: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SpeakerTurn]:
        """Return speaker turns for normalized audio samples."""
        if sample_rate != NORMALIZED_SAMPLE_RATE:
            raise DiarizationProcessingError(
                "Speaker diarization expects normalized 16000 Hz audio."
            )

        sherpa_onnx = _load_sherpa_onnx()
        if sherpa_onnx is None:
            raise DiarizationProcessingError("sherpa-onnx is unavailable for speaker diarization.")

        paths = ensure_default_models(self._cache_dir)
        turns = self._run_diarization(
            sherpa_onnx,
            samples,
            paths=paths,
            num_clusters=-1,
            progress_callback=progress_callback,
        )
        if _speaker_count(turns) > max_speakers:
            turns = self._run_diarization(
                sherpa_onnx,
                samples,
                paths=paths,
                num_clusters=max_speakers,
                progress_callback=progress_callback,
            )
        return normalize_speaker_labels(turns)

    def _run_diarization(
        self,
        sherpa_onnx: ModuleType,
        samples: np.ndarray,
        *,
        paths: DiarizationModelPaths,
        num_clusters: int,
        progress_callback: Callable[[int, int], None] | None,
    ) -> list[SpeakerTurn]:
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(paths.segmentation_model)
                )
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(paths.embedding_model)),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_clusters, threshold=DEFAULT_CLUSTER_THRESHOLD
            ),
            min_duration_on=DEFAULT_MIN_DURATION_ON_SEC,
            min_duration_off=DEFAULT_MIN_DURATION_OFF_SEC,
        )
        if not config.validate():
            raise DiarizationProcessingError("Speaker diarization model configuration is invalid.")

        try:
            diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
            result = diarizer.process(samples, callback=_sherpa_progress(progress_callback))
        except RuntimeError as error:
            raise DiarizationProcessingError(str(error)) from error

        return [
            SpeakerTurn(
                start_sec=float(item.start),
                end_sec=float(item.end),
                speaker=str(item.speaker),
            )
            for item in result.sort_by_start_time()
            if float(item.end) > float(item.start)
        ]


def ensure_default_models(cache_dir: Path | None = None) -> DiarizationModelPaths:
    """Download and verify default diarization models when missing."""
    paths = default_model_paths(cache_dir)
    _ensure_segmentation_model(paths.segmentation_model)
    _ensure_file(
        paths.embedding_model,
        url=EMBEDDING_MODEL_URL,
        expected_sha256=EMBEDDING_MODEL_SHA256,
    )
    return paths


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


def _speaker_count(turns: list[SpeakerTurn]) -> int:
    return len({turn.speaker for turn in turns})


def _ensure_segmentation_model(model_path: Path) -> None:
    if _verified(model_path, SEGMENTATION_MODEL_SHA256):
        return

    archive_path = model_path.parent.parent / f"{SEGMENTATION_MODEL_DIR}.tar.bz2"
    _ensure_file(
        archive_path,
        url=SEGMENTATION_ARCHIVE_URL,
        expected_sha256=SEGMENTATION_ARCHIVE_SHA256,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:bz2") as archive:
        member = archive.getmember(f"{SEGMENTATION_MODEL_DIR}/{SEGMENTATION_MODEL_FILE}")
        archive.extract(member, path=model_path.parent.parent, filter="data")

    if not _verified(model_path, SEGMENTATION_MODEL_SHA256):
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
    except ImportError:
        return None
