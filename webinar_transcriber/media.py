"""Media probing helpers."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, overload

import av

from webinar_transcriber.models import AudioAsset, MediaAsset, VideoAsset

if TYPE_CHECKING:
    from av.audio.stream import AudioStream
    from av.container import InputContainer, OutputContainer
    from av.video.stream import VideoStream


class MediaProcessingError(RuntimeError):
    """Raised when media work cannot be completed."""


@overload
def _first_input_stream(
    input_container: "InputContainer", stream_type: Literal["audio"]
) -> "AudioStream | None": ...


@overload
def _first_input_stream(
    input_container: "InputContainer", stream_type: Literal["video"]
) -> "VideoStream | None": ...


def _first_input_stream(
    input_container: "InputContainer", stream_type: Literal["audio", "video"]
) -> "AudioStream | VideoStream | None":
    return cast(
        "AudioStream | VideoStream | None",
        next((stream for stream in input_container.streams if stream.type == stream_type), None),
    )


@overload
def _required_input_stream(
    input_container: "InputContainer",
    stream_type: Literal["audio"],
    *,
    error_message: str,
) -> "AudioStream": ...


@overload
def _required_input_stream(
    input_container: "InputContainer",
    stream_type: Literal["video"],
    *,
    error_message: str,
) -> "VideoStream": ...


def _required_input_stream(
    input_container: "InputContainer",
    stream_type: Literal["audio", "video"],
    *,
    error_message: str,
) -> "AudioStream | VideoStream":
    stream = _first_input_stream(input_container, stream_type)
    if stream is None:
        raise MediaProcessingError(error_message)
    return stream


@contextmanager
def open_input_media_container(
    path: Path,
    *,
    error_message: str,
) -> Iterator["InputContainer"]:
    """Open a PyAV input container and normalize open-time failures."""
    try:
        with av.open(str(path), mode="r") as container:
            yield container
    except (FileNotFoundError, OSError, av.FFmpegError) as error:
        raise MediaProcessingError(error_message.format(path=path, error=error)) from error


@contextmanager
def open_output_media_container(
    path: Path,
    *,
    error_message: str,
) -> Iterator["OutputContainer"]:
    """Open a PyAV output container and normalize open-time failures."""
    try:
        with av.open(str(path), mode="w") as container:
            yield container
    except (FileNotFoundError, OSError, av.FFmpegError) as error:
        raise MediaProcessingError(error_message.format(path=path, error=error)) from error


def _pyav_stream_has_attached_picture(stream: object) -> bool:
    disposition = getattr(stream, "disposition", None)
    attached_pic_flag = getattr(disposition, "attached_pic", None)
    if disposition is None or attached_pic_flag is None:
        return False
    return bool(disposition & attached_pic_flag)


def _stream_duration_sec(stream: object) -> float | None:
    duration = getattr(stream, "duration", None)
    time_base = getattr(stream, "time_base", None)
    if duration is None or time_base is None:
        return None
    return float(duration * time_base)


def probe_media(input_path: Path) -> MediaAsset:
    """Inspect media metadata with PyAV and return normalized stream details.

    Returns:
        MediaAsset: The normalized probed media metadata.

    Raises:
        MediaProcessingError: If the input contains no usable audio or video streams.
    """
    with open_input_media_container(
        input_path,
        error_message="Could not open {path} with PyAV: {error}",
    ) as input_container:
        streams = list(input_container.streams)
        audio_stream = _first_input_stream(input_container, "audio")
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
        duration_sec = (
            float(input_container.duration / av.time_base)
            if input_container.duration is not None
            else (audio_duration or 0.0)
        )

        audio_codec_context = getattr(audio_stream, "codec_context", None)
        parsed_sample_rate = (
            int(sample_rate)
            if audio_stream is not None
            and (sample_rate := getattr(audio_codec_context, "sample_rate", None)) is not None
            else None
        )
        parsed_channels = (
            int(channels)
            if audio_stream is not None
            and (channels := getattr(audio_codec_context, "channels", None)) is not None
            else None
        )

        if video_stream is None:
            return AudioAsset(
                path=str(input_path),
                duration_sec=duration_sec,
                sample_rate=parsed_sample_rate,
                channels=parsed_channels,
            )

        video_codec_context = getattr(video_stream, "codec_context", None)
        width = getattr(video_codec_context, "width", None)
        height = getattr(video_codec_context, "height", None)
        return VideoAsset(
            path=str(input_path),
            duration_sec=duration_sec,
            sample_rate=parsed_sample_rate,
            channels=parsed_channels,
            fps=float(video_stream.average_rate) if video_stream.average_rate is not None else None,
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
        )
