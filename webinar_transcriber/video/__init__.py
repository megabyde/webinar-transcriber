"""Video processing helpers."""

from __future__ import annotations

from webinar_transcriber.video.frames import extract_representative_frames
from webinar_transcriber.video.scenes import detect_scenes

__all__ = ["detect_scenes", "extract_representative_frames"]
