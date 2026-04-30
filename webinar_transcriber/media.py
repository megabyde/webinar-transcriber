"""Media probing helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import av

from webinar_transcriber.models import AudioAsset, MediaAsset, VideoAsset

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from av.audio.stream import AudioStream
    from av.container import InputContainer, OutputContainer
    from av.video.stream import VideoStream


class MediaProcessingError(RuntimeError):
    """Raised when media work cannot be completed."""


def _first_audio_stream(input_container: InputContainer) -> AudioStream | None:
    stream = next((stream for stream in input_container.streams if stream.type == "audio"), None)
    return cast("AudioStream | None", stream)


def _first_video_stream(input_container: InputContainer) -> VideoStream | None:
    stream = next((stream for stream in input_container.streams if stream.type == "video"), None)
    return cast("VideoStream | None", stream)


def _required_audio_stream(input_container: InputContainer, *, error_message: str) -> AudioStream:
    stream = _first_audio_stream(input_container)
    if stream is None:
        raise MediaProcessingError(error_message)
    return stream


def _required_video_stream(input_container: InputContainer, *, error_message: str) -> VideoStream:
    stream = _first_video_stream(input_container)
    if stream is None:
        raise MediaProcessingError(error_message)
    return stream


@contextmanager
def open_input_media_container(path: Path, *, error_message: str) -> Iterator[InputContainer]:
    """Open a PyAV input container and normalize open-time failures."""
    try:
        with av.open(str(path), mode="r") as container:
            yield container
    except (FileNotFoundError, OSError, av.FFmpegError) as error:
        raise MediaProcessingError(error_message.format(path=path, error=error)) from error


@contextmanager
def open_output_media_container(path: Path, *, error_message: str) -> Iterator[OutputContainer]:
    """Open a PyAV output container and normalize open-time failures."""
    try:
        with av.open(str(path), mode="w") as container:
            yield container
    except (FileNotFoundError, OSError, av.FFmpegError) as error:
        raise MediaProcessingError(error_message.format(path=path, error=error)) from error


def _pyav_stream_has_attached_picture(stream: object) -> bool:
    typed_stream = cast("Any", stream)
    return bool(typed_stream.disposition & typed_stream.disposition.attached_pic)


def _stream_duration_sec(stream: object) -> float | None:
    timed_stream = cast("Any", stream)
    if timed_stream.duration is None or timed_stream.time_base is None:
        return None
    return float(timed_stream.duration * timed_stream.time_base)


def probe_media(input_path: Path) -> MediaAsset:
    """Inspect media metadata with PyAV and return normalized stream details.

    Returns:
        MediaAsset: The normalized probed media metadata.

    Raises:
        MediaProcessingError: If the input contains no usable audio or video streams.
    """
    with open_input_media_container(
        input_path, error_message="Could not open {path} with PyAV: {error}"
    ) as input_container:
        streams = list(input_container.streams)
        audio_stream = _first_audio_stream(input_container)
        video_stream = next(
            (
                stream
                for stream in streams
                if stream.type == "video" and not _pyav_stream_has_attached_picture(stream)
            ),
            None,
        )

        if audio_stream is None and video_stream is None:
            raise MediaProcessingError(f"No audio or video stream found in {input_path}.")

        audio_duration = _stream_duration_sec(audio_stream) if audio_stream is not None else None
        container_duration = input_container.duration
        duration_sec = audio_duration or 0.0
        if container_duration is not None:
            duration_sec = float(container_duration / av.time_base)

        parsed_sample_rate = None
        parsed_channels = None
        if audio_stream is not None:
            sample_rate = audio_stream.codec_context.sample_rate
            channels = audio_stream.codec_context.channels
            parsed_sample_rate = int(sample_rate) if sample_rate is not None else None
            parsed_channels = int(channels) if channels is not None else None

        if video_stream is None:
            return AudioAsset(
                path=str(input_path),
                duration_sec=duration_sec,
                sample_rate=parsed_sample_rate,
                channels=parsed_channels,
            )

        video_codec_context = cast("Any", video_stream.codec_context)
        width = video_codec_context.width
        height = video_codec_context.height
        return VideoAsset(
            path=str(input_path),
            duration_sec=duration_sec,
            sample_rate=parsed_sample_rate,
            channels=parsed_channels,
            fps=float(video_stream.average_rate) if video_stream.average_rate is not None else None,
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
        )
