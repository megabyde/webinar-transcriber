"""Video processing helpers."""

from webinar_transcriber.video.frames import extract_representative_frames
from webinar_transcriber.video.scenes import detect_scenes, estimate_sample_count

__all__ = ["detect_scenes", "estimate_sample_count", "extract_representative_frames"]
