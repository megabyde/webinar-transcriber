"""Media probing helpers and shared command execution."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import cast

from webinar_transcriber.models import AudioAsset, MediaAsset, VideoAsset

MEDIA_COMMAND_TIMEOUT_SEC = 300.0


class MediaProcessingError(RuntimeError):
    """Raised when ffmpeg or ffprobe work cannot be completed."""


def run_media_command(
    *args: str, timeout_sec: float = MEDIA_COMMAND_TIMEOUT_SEC
) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg or ffprobe and normalize common failure modes.

    Returns:
        subprocess.CompletedProcess[str]: The completed subprocess result.

    Raises:
        MediaProcessingError: If the command fails or times out.
    """
    try:
        result = subprocess.run(
            args, capture_output=True, check=True, text=True, timeout=timeout_sec
        )
    except subprocess.CalledProcessError as ex:
        raise MediaProcessingError(ex.stderr.strip() or "External media command failed.") from ex
    except subprocess.TimeoutExpired as ex:
        command_name = Path(args[0]).name if args else "External media command"
        raise MediaProcessingError(f"{command_name} timed out after {timeout_sec:g}s.") from ex
    return result


def _parse_frame_rate(raw_value: str | None) -> float | None:
    if not raw_value or raw_value == "0/0":
        return None
    return float(Fraction(raw_value))


def _is_attached_picture_stream(stream: dict[str, object]) -> bool:
    disposition = stream.get("disposition")
    if not isinstance(disposition, dict):
        return False
    return bool(cast("dict[str, object]", disposition).get("attached_pic"))


def probe_media(input_path: Path) -> MediaAsset:
    """Inspect media with ffprobe and return normalized metadata.

    Returns:
        MediaAsset: The normalized probed media metadata.

    Raises:
        MediaProcessingError: If the input contains no usable audio or video streams.
    """
    result = run_media_command(
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(input_path),
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    video_stream = next(
        (
            stream
            for stream in streams
            if stream.get("codec_type") == "video" and not _is_attached_picture_stream(stream)
        ),
        None,
    )

    if audio_stream is None and video_stream is None:
        raise MediaProcessingError(f"No audio or video stream found in {input_path}.")

    audio_duration = None
    parsed_sample_rate = None
    parsed_channels = None
    if audio_stream is not None:
        audio_duration = audio_stream.get("duration")
        if sample_rate := audio_stream.get("sample_rate"):
            parsed_sample_rate = int(sample_rate)
        if channels := audio_stream.get("channels"):
            parsed_channels = int(channels)

    duration_raw = payload.get("format", {}).get("duration") or audio_duration or "0"
    duration_sec = float(duration_raw)

    if video_stream is None:
        return AudioAsset(
            path=str(input_path),
            duration_sec=duration_sec,
            sample_rate=parsed_sample_rate,
            channels=parsed_channels,
        )

    return VideoAsset(
        path=str(input_path),
        duration_sec=duration_sec,
        sample_rate=parsed_sample_rate,
        channels=parsed_channels,
        fps=_parse_frame_rate(video_stream.get("avg_frame_rate")),
        width=int(video_stream["width"]) if video_stream.get("width") else None,
        height=int(video_stream["height"]) if video_stream.get("height") else None,
    )
