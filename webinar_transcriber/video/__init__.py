"""Video processing helpers."""

from __future__ import annotations

from webinar_transcriber.video.scenes import detect_scenes, estimated_scene_sample_count

__all__ = [
    "detect_scenes",
    "estimated_scene_sample_count",
]
