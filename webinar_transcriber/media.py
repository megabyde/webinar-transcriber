"""Media probing and normalization helpers."""

import json
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

from webinar_transcriber.models import MediaAsset, MediaType


class MediaProcessingError(RuntimeError):
    """Raised when ffmpeg or ffprobe work cannot be completed."""


def _run_command(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        raise MediaProcessingError(result.stderr.strip() or "External media command failed.")
    return result


def _parse_frame_rate(raw_value: str | None) -> float | None:
    if not raw_value or raw_value == "0/0":
        return None
    return float(Fraction(raw_value))


def probe_media(input_path: Path) -> MediaAsset:
    """Inspect media with ffprobe and return normalized metadata."""
    result = _run_command(
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
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)

    if audio_stream is None and video_stream is None:
        raise MediaProcessingError(f"No audio or video stream found in {input_path}.")

    media_type = MediaType.VIDEO if video_stream is not None else MediaType.AUDIO
    audio_duration = audio_stream.get("duration") if audio_stream else None
    duration_raw = payload.get("format", {}).get("duration") or audio_duration or "0"
    sample_rate = audio_stream.get("sample_rate") if audio_stream else None
    channels = audio_stream.get("channels") if audio_stream else None

    return MediaAsset(
        path=str(input_path),
        media_type=media_type,
        duration_sec=float(duration_raw),
        sample_rate=int(sample_rate) if sample_rate else None,
        channels=int(channels) if channels else None,
        fps=_parse_frame_rate(video_stream.get("avg_frame_rate")) if video_stream else None,
        width=int(video_stream["width"]) if video_stream and video_stream.get("width") else None,
        height=int(video_stream["height"]) if video_stream and video_stream.get("height") else None,
    )


def extract_audio(input_path: Path, output_path: Path) -> Path:
    """Convert the input media into a mono 16 kHz WAV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    )
    return output_path


@contextmanager
def prepared_transcription_audio(input_path: Path, media_asset: MediaAsset) -> Iterator[Path]:
    """Yield a transcription-ready audio path and clean up temp files when needed."""
    if media_asset.media_type is MediaType.AUDIO:
        yield input_path
        return

    with tempfile.TemporaryDirectory(prefix="webinar-transcriber-audio-") as temp_dir:
        audio_path = Path(temp_dir) / f"{input_path.stem}.wav"
        extract_audio(input_path, audio_path)
        yield audio_path
