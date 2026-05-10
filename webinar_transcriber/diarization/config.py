"""Default local speaker-diarization model configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DIARIZATION_CACHE_ENV_VAR = "WEBINAR_DIARIZATION_CACHE_DIR"
DEFAULT_MAX_SPEAKERS = 6
DEFAULT_CLUSTER_THRESHOLD = 0.5
DEFAULT_MIN_DURATION_ON_SEC = 0.3
DEFAULT_MIN_DURATION_OFF_SEC = 0.5
DEFAULT_DIARIZATION_MODEL = "pyannote-segmentation-3.0-fp32+nemo-titanet-small"

SEGMENTATION_MODEL_DIR = "sherpa-onnx-pyannote-segmentation-3-0"
SEGMENTATION_MODEL_FILE = "model.onnx"
EMBEDDING_MODEL_FILE = "nemo_en_titanet_small.onnx"

SEGMENTATION_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
SEGMENTATION_ARCHIVE_SHA256 = "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488"
SEGMENTATION_MODEL_SHA256 = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"

EMBEDDING_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/nemo_en_titanet_small.onnx"
)
EMBEDDING_MODEL_SHA256 = "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"


@dataclass(frozen=True)
class DiarizationModelPaths:
    """Resolved local model paths for sherpa-onnx diarization."""

    segmentation_model: Path
    embedding_model: Path


def default_cache_dir() -> Path:
    """Return the speaker-diarization model cache directory."""
    configured = os.environ.get(DIARIZATION_CACHE_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "webinar-transcriber" / "diarization"


def default_model_paths(cache_dir: Path | None = None) -> DiarizationModelPaths:
    """Return expected local model paths under the configured cache directory."""
    root = cache_dir or default_cache_dir()
    return DiarizationModelPaths(
        segmentation_model=root / SEGMENTATION_MODEL_DIR / SEGMENTATION_MODEL_FILE,
        embedding_model=root / EMBEDDING_MODEL_FILE,
    )
