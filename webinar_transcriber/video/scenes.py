"""Scene detection for webinar videos."""

from pathlib import Path

import av
import numpy as np

from webinar_transcriber.models import Scene


def detect_scenes(video_path: Path) -> list[Scene]:
    """Detect scenes in a video file using simple frame-difference scoring."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        duration_sec = (
            float(stream.duration) * float(stream.time_base)
            if stream.duration is not None and stream.time_base is not None
            else 0.0
        )
        scene_starts = [0.0]
        previous_frame: np.ndarray | None = None
        previous_time = 0.0

        for frame in container.decode(stream):
            current_time = float(frame.time or 0.0)
            current_rgb = frame.to_ndarray(format="rgb24").astype(np.float32)

            if previous_frame is not None:
                difference = np.abs(current_rgb - previous_frame).mean()
                long_enough = (current_time - scene_starts[-1]) >= 0.5
                if difference >= 18.0 and long_enough:
                    scene_starts.append(current_time)

            previous_frame = current_rgb
            previous_time = current_time

        if duration_sec <= 0.0:
            duration_sec = previous_time

    if not scene_starts:
        scene_starts = [0.0]

    scene_bounds = list(zip(scene_starts, [*scene_starts[1:], duration_sec], strict=False))

    return [
        Scene(id=f"scene-{index}", start_sec=float(start), end_sec=float(end))
        for index, (start, end) in enumerate(scene_bounds, start=1)
    ]
