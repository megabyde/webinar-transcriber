"""Scene detection for webinar videos."""

from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video

from webinar_transcriber.models import Scene


def detect_scenes(video_path: Path) -> list[Scene]:
    """Detect scenes in a video file using content changes."""
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=12.0, min_scene_len=3))
    video = open_video(str(video_path))
    scene_manager.detect_scenes(video=video)
    scene_boundaries = scene_manager.get_scene_list()

    if not scene_boundaries:
        duration = video.duration.get_seconds() if video.duration is not None else 0.0
        return [Scene(id="scene-1", start_sec=0.0, end_sec=float(duration))]

    return [
        Scene(
            id=f"scene-{index}",
            start_sec=float(start.get_seconds()),
            end_sec=float(end.get_seconds()),
        )
        for index, (start, end) in enumerate(scene_boundaries, start=1)
    ]
