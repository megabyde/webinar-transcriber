"""Media probing helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, cast

import av
from av.error import FFmpegError

from webinar_transcriber.models import AudioAsset, MediaAsset, VideoAsset

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from av.audio.stream import AudioStream
    from av.container import InputContainer, OutputContainer
    from av.video.stream import VideoStream


class _HasDuration(Protocol):
    duration: Any
    time_base: Any


class MediaProcessingError(RuntimeError):
    """Raised when media work cannot be completed."""


def _first_audio_stream(input_container: InputContainer) -> AudioStream | None:
    stream = next((stream for stream in input_container.streams if stream.type == "audio"), None)
    # PyAV stubs expose .streams as base Stream; filtering on .type does not narrow it
    return cast("AudioStream | None", stream)


def _first_video_stream(input_container: InputContainer) -> VideoStream | None:
    # Cover art is exposed as a video stream; only a moving-picture stream counts as video
    stream = next(
        (
            stream
            for stream in input_container.streams
            if stream.type == "video" and not stream.disposition & stream.disposition.attached_pic
        ),
        None,
    )
    # PyAV stubs expose .streams as base Stream; filtering on .type does not narrow it
    return cast("VideoStream | None", stream)


@contextmanager
def open_input_media_container(path: Path) -> Generator[InputContainer, None, None]:
    """Open a PyAV input container and normalize open-time failures."""
    try:
        input_container = av.open(str(path), mode="r")
    except (FileNotFoundError, OSError, FFmpegError) as ex:
        raise MediaProcessingError(f"Could not open {path} with PyAV: {ex}") from ex
    with input_container as container:
        yield container


@contextmanager
def open_audio_input_container(
    path: Path,
) -> Generator[tuple[InputContainer, AudioStream], None, None]:
    """Open a PyAV input container and yield its required audio stream."""
    with open_input_media_container(path) as container:
        stream = _first_audio_stream(container)
        if stream is None:
            raise MediaProcessingError(f"No audio stream found in {path}.")
        yield container, stream


@contextmanager
def open_video_input_container(
    path: Path,
) -> Generator[tuple[InputContainer, VideoStream], None, None]:
    """Open a PyAV input container and yield its required video stream."""
    with open_input_media_container(path) as container:
        stream = _first_video_stream(container)
        if stream is None:
            raise MediaProcessingError(f"No video stream found in {path}.")
        yield container, stream


@contextmanager
def open_output_media_container(path: Path) -> Generator[OutputContainer, None, None]:
    """Open a PyAV output container and normalize open-time failures."""
    try:
        output_container = av.open(str(path), mode="w")
    except (FileNotFoundError, OSError, FFmpegError) as ex:
        raise MediaProcessingError(f"Could not open {path} for writing with PyAV: {ex}") from ex
    with output_container as container:
        yield container


def _stream_duration_sec(stream: _HasDuration) -> float | None:
    if stream.duration is None or not stream.time_base:
        return None
    return float(stream.duration * stream.time_base)


def probe_media(input_path: Path) -> MediaAsset:
    """Inspect media metadata with PyAV and return normalized stream details.

    Returns:
        MediaAsset: The normalized probed media metadata.

    Raises:
        MediaProcessingError: If the input contains no usable audio stream.
    """
    with open_input_media_container(input_path) as input_container:
        audio_stream = _first_audio_stream(input_container)
        video_stream = _first_video_stream(input_container)

        if audio_stream is None:
            raise MediaProcessingError(f"No audio stream found in {input_path}.")

        container_duration = input_container.duration
        audio_duration = _stream_duration_sec(audio_stream)
        video_duration = _stream_duration_sec(video_stream) if video_stream is not None else None
        if container_duration is not None:
            duration_sec = float(container_duration / av.time_base)
        elif audio_duration is not None:
            duration_sec = audio_duration
        elif video_duration is not None:
            duration_sec = video_duration
        else:
            duration_sec = 0.0

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

        video_codec_context = video_stream.codec_context
        width = video_codec_context.width
        height = video_codec_context.height
        return VideoAsset(
            path=str(input_path),
            duration_sec=duration_sec,
            sample_rate=parsed_sample_rate,
            channels=parsed_channels,
            fps=float(video_stream.average_rate) if video_stream.average_rate else None,
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
        )
