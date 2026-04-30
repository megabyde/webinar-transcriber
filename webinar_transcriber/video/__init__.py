"""Video processing helpers."""

from __future__ import annotations

from webinar_transcriber.video.frames import extract_representative_frames
from webinar_transcriber.video.scenes import (
    DEFAULT_SCENE_DETECTION_SETTINGS,
    SCENE_SCAN_FPS,
    SceneDetectionSettings,
    detect_scenes,
    estimated_scene_sample_count,
)

__all__ = [
    "DEFAULT_SCENE_DETECTION_SETTINGS",
    "SCENE_SCAN_FPS",
    "SceneDetectionSettings",
    "detect_scenes",
    "estimated_scene_sample_count",
    "extract_representative_frames",
]
