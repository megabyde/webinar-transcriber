"""Validation and copying for persisted report-replay inputs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from webinar_transcriber.models import (
    AudioAsset,
    MediaAsset,
    MediaType,
    Scene,
    TranscriptionResult,
    TranscriptSegment,
    VideoAsset,
)

if TYPE_CHECKING:
    from webinar_transcriber.paths import RunLayout


class ReplayValidationError(ValueError):
    """Raised when a persisted run cannot be replayed safely."""


_TIMELINE_DURATION_TOLERANCE_SEC = 1.0


@dataclass(slots=True, frozen=True)
class ReplaySource:
    """Validated persisted inputs needed to rebuild a report."""

    run_dir: Path
    media_asset: MediaAsset
    transcription: TranscriptionResult
    scenes: list[Scene]
    frame_paths: list[PurePosixPath]
    diarization_path: Path | None


def load_replay_source(run_dir: Path) -> ReplaySource:
    """Validate and load the persisted inputs required for report replay.

    Returns:
        ReplaySource: Typed replay inputs and their validated source paths.
    """
    run_dir = run_dir.resolve()
    media_asset = _load_media_asset(_read_json(run_dir / "metadata.json"))
    transcription = _load_transcription(
        _read_json(run_dir / "transcript.json"), duration_sec=media_asset.duration_sec
    )

    scenes: list[Scene] = []
    frame_paths: list[PurePosixPath] = []
    if isinstance(media_asset, VideoAsset):
        scenes, frame_paths = _load_scenes(
            _read_json(run_dir / "scenes.json"),
            run_dir=run_dir,
            duration_sec=media_asset.duration_sec,
        )

    diarization_path = run_dir / "diarization.json"
    if diarization_path.exists():
        _validate_diarization(_read_json(diarization_path), media_asset.duration_sec)
    else:
        diarization_path = None

    return ReplaySource(
        run_dir=run_dir,
        media_asset=media_asset,
        transcription=transcription,
        scenes=scenes,
        frame_paths=frame_paths,
        diarization_path=diarization_path,
    )


def copy_replay_source(source: ReplaySource, layout: RunLayout) -> None:
    """Copy validated replay inputs into a new self-contained run directory."""
    try:
        shutil.copy2(source.run_dir / "metadata.json", layout.metadata_path)
        shutil.copy2(source.run_dir / "transcript.json", layout.transcript_path)
        if isinstance(source.media_asset, VideoAsset):
            shutil.copy2(source.run_dir / "scenes.json", layout.scenes_path)
            for relative_path in source.frame_paths:
                destination = layout.run_dir.joinpath(*relative_path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source.run_dir.joinpath(*relative_path.parts), destination)
        if source.diarization_path is not None:
            shutil.copy2(source.diarization_path, layout.diarization_path)
    except OSError as ex:
        raise ReplayValidationError(f"Could not copy replay artifacts: {ex}") from ex


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as ex:
        raise ReplayValidationError(
            f"Replay source is missing required artifact: {path.name}"
        ) from ex
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise ReplayValidationError(f"Replay source has invalid JSON in {path.name}: {ex}") from ex


def _load_media_asset(payload: object) -> MediaAsset:
    data = _object(payload, "metadata.json")
    media_type = _string(data.get("media_type"), "metadata.json.media_type")
    common_keys = {"path", "duration_sec", "sample_rate", "channels", "media_type"}
    if media_type == MediaType.AUDIO:
        _keys(data, common_keys, artifact="metadata.json")
        return AudioAsset(
            path=_string(data["path"], "metadata.json.path"),
            duration_sec=_nonnegative_number(data["duration_sec"], "metadata.json.duration_sec"),
            sample_rate=_optional_positive_integer(
                data["sample_rate"], "metadata.json.sample_rate"
            ),
            channels=_optional_positive_integer(data["channels"], "metadata.json.channels"),
        )
    if media_type == MediaType.VIDEO:
        _keys(data, common_keys | {"fps", "width", "height"}, artifact="metadata.json")
        return VideoAsset(
            path=_string(data["path"], "metadata.json.path"),
            duration_sec=_nonnegative_number(data["duration_sec"], "metadata.json.duration_sec"),
            sample_rate=_optional_positive_integer(
                data["sample_rate"], "metadata.json.sample_rate"
            ),
            channels=_optional_positive_integer(data["channels"], "metadata.json.channels"),
            fps=_optional_positive_number(data["fps"], "metadata.json.fps"),
            width=_optional_positive_integer(data["width"], "metadata.json.width"),
            height=_optional_positive_integer(data["height"], "metadata.json.height"),
        )
    raise ReplayValidationError(
        "Replay source field metadata.json.media_type must be 'audio' or 'video'."
    )


def _load_transcription(payload: object, *, duration_sec: float) -> TranscriptionResult:
    data = _object(payload, "transcript.json")
    _keys(data, {"detected_language", "segments"}, artifact="transcript.json")
    detected_language = _optional_string(
        data["detected_language"], "transcript.json.detected_language"
    )
    segment_payloads = _array(data["segments"], "transcript.json.segments")
    segments: list[TranscriptSegment] = []
    ids: set[str] = set()
    for index, payload_segment in enumerate(segment_payloads):
        label = f"transcript.json.segments[{index}]"
        segment = _object(payload_segment, label)
        required_keys = {"id", "start_sec", "end_sec", "text"}
        _keys(segment, required_keys, optional={"speaker"}, artifact=label)
        segment_id = _string(segment["id"], f"{label}.id")
        if segment_id in ids:
            raise ReplayValidationError(
                f"Replay source contains duplicate segment id: {segment_id}"
            )
        ids.add(segment_id)
        start_sec, end_sec = _timeline(segment, label, duration_sec=duration_sec)
        segments.append(
            TranscriptSegment(
                id=segment_id,
                start_sec=start_sec,
                end_sec=end_sec,
                text=_string(segment["text"], f"{label}.text", allow_empty=True),
                speaker=_optional_string(segment.get("speaker"), f"{label}.speaker"),
            )
        )
    return TranscriptionResult(detected_language=detected_language, segments=segments)


def _load_scenes(
    payload: object, *, run_dir: Path, duration_sec: float
) -> tuple[list[Scene], list[PurePosixPath]]:
    scene_payloads = _array(payload, "scenes.json")
    scenes: list[Scene] = []
    frame_paths: list[PurePosixPath] = []
    ids: set[str] = set()
    paths: set[PurePosixPath] = set()
    for index, payload_scene in enumerate(scene_payloads):
        label = f"scenes.json[{index}]"
        data = _object(payload_scene, label)
        _keys(data, {"id", "start_sec", "end_sec", "image_path"}, artifact=label)
        scene_id = _string(data["id"], f"{label}.id")
        if scene_id in ids:
            raise ReplayValidationError(f"Replay source contains duplicate scene id: {scene_id}")
        ids.add(scene_id)
        start_sec, end_sec = _timeline(data, label, duration_sec=duration_sec)
        image_path = _string(data["image_path"], f"{label}.image_path")
        relative_path = PurePosixPath(image_path)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative_path.parts
            or relative_path.parts[0] != "frames"
        ):
            raise ReplayValidationError(
                f"Replay source frame path must stay under frames/: {image_path}"
            )
        if relative_path in paths:
            raise ReplayValidationError(
                f"Replay source contains duplicate frame path: {image_path}"
            )
        paths.add(relative_path)
        frame_path = run_dir.joinpath(*relative_path.parts)
        resolved_frame_path = frame_path.resolve()
        if not resolved_frame_path.is_relative_to(run_dir):
            raise ReplayValidationError(
                f"Replay source frame path must stay under frames/: {image_path}"
            )
        if not resolved_frame_path.is_file():
            raise ReplayValidationError(f"Replay source is missing referenced frame: {image_path}")
        frame_paths.append(relative_path)
        scenes.append(
            Scene(
                id=scene_id,
                start_sec=start_sec,
                end_sec=end_sec,
                image_path=image_path,
            )
        )
    return scenes, frame_paths


def _validate_diarization(payload: object, duration_sec: float) -> None:
    turns = _array(payload, "diarization.json")
    for index, payload_turn in enumerate(turns):
        label = f"diarization.json[{index}]"
        turn = _object(payload_turn, label)
        _keys(turn, {"start_sec", "end_sec", "speaker"}, artifact=label)
        _timeline(turn, label, duration_sec=duration_sec)
        _string(turn["speaker"], f"{label}.speaker")


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ReplayValidationError(f"Replay source field {label} must be an object.")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ReplayValidationError(f"Replay source field {label} must be an array.")
    items: list[object] = list(value)
    return items


def _keys(
    data: dict[str, object],
    required: set[str],
    *,
    artifact: str,
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    if missing := sorted(required - data.keys()):
        raise ReplayValidationError(
            f"Replay source field {artifact} is missing keys: {', '.join(missing)}"
        )
    if unknown := sorted(data.keys() - allowed):
        raise ReplayValidationError(
            f"Replay source field {artifact} has unknown keys: {', '.join(unknown)}"
        )


def _string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ReplayValidationError(f"Replay source field {label} must be a string.")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayValidationError(f"Replay source field {label} must be a number.")
    return float(value)


def _positive_number(value: object, label: str) -> float:
    number = _number(value, label)
    if number <= 0:
        raise ReplayValidationError(f"Replay source field {label} must be positive.")
    return number


def _nonnegative_number(value: object, label: str) -> float:
    number = _number(value, label)
    if number < 0:
        raise ReplayValidationError(f"Replay source field {label} must be nonnegative.")
    return number


def _optional_positive_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    return _positive_number(value, label)


def _optional_positive_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReplayValidationError(f"Replay source field {label} must be a positive integer.")
    return value


def _timeline(data: dict[str, object], label: str, *, duration_sec: float) -> tuple[float, float]:
    start_sec = _number(data["start_sec"], f"{label}.start_sec")
    end_sec = _number(data["end_sec"], f"{label}.end_sec")
    exceeds_known_duration = (
        duration_sec > 0 and end_sec > duration_sec + _TIMELINE_DURATION_TOLERANCE_SEC
    )
    if start_sec < 0 or end_sec < start_sec or exceeds_known_duration:
        raise ReplayValidationError(
            f"Replay source field {label} has invalid bounds: {start_sec:g}-{end_sec:g}"
        )
    return start_sec, end_sec
