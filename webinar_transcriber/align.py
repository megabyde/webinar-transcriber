"""Alignment helpers for transcript segments and video scenes."""

import re

from rapidfuzz import fuzz

from webinar_transcriber.models import (
    AlignmentBlock,
    OcrResult,
    Scene,
    SlideFrame,
    TranscriptSegment,
)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\u0400-\u04FF]+")


def align_by_time(
    transcript_segments: list[TranscriptSegment],
    scenes: list[Scene],
    slide_frames: list[SlideFrame],
) -> list[AlignmentBlock]:
    """Assign transcript segments to scenes using midpoint inclusion."""
    frame_by_scene = {frame.scene_id: frame for frame in slide_frames}
    blocks: list[AlignmentBlock] = []

    for index, scene in enumerate(scenes, start=1):
        scene_segments = [
            segment
            for segment in transcript_segments
            if scene.start_sec <= ((segment.start_sec + segment.end_sec) / 2) < scene.end_sec
        ]
        transcript_text = " ".join(segment.text for segment in scene_segments).strip()
        frame = frame_by_scene.get(scene.id)
        blocks.append(
            AlignmentBlock(
                id=f"block-{index}",
                start_sec=scene.start_sec,
                end_sec=scene.end_sec,
                transcript_segment_ids=[segment.id for segment in scene_segments],
                transcript_text=transcript_text,
                scene_id=scene.id,
                frame_id=frame.id if frame else None,
            )
        )

    return blocks


def align_with_ocr(
    transcript_segments: list[TranscriptSegment],
    scenes: list[Scene],
    slide_frames: list[SlideFrame],
    ocr_results: list[OcrResult],
) -> list[AlignmentBlock]:
    """Assign transcript segments to scenes using OCR and time proximity."""
    frame_by_scene = {frame.scene_id: frame for frame in slide_frames}
    ocr_by_frame = {
        result.frame_id: result for result in ocr_results if _is_informative_ocr(result)
    }

    if not ocr_by_frame:
        return align_by_time(transcript_segments, scenes, slide_frames)

    assignments: dict[str, list[tuple[TranscriptSegment, float]]] = {
        scene.id: [] for scene in scenes
    }

    for segment in transcript_segments:
        best_scene = None
        best_score = -1.0

        for scene in scenes:
            frame = frame_by_scene.get(scene.id)
            ocr_result = ocr_by_frame.get(frame.id) if frame else None
            time_score = _time_score(segment, scene)
            text_score = _text_score(segment.text, ocr_result.text if ocr_result else "")
            final_score = (0.4 * time_score) + (0.6 * text_score)

            if final_score > best_score:
                best_scene = scene
                best_score = final_score

        if best_scene is not None:
            assignments[best_scene.id].append((segment, best_score))

    blocks: list[AlignmentBlock] = []

    for index, scene in enumerate(scenes, start=1):
        frame = frame_by_scene.get(scene.id)
        ocr_result = ocr_by_frame.get(frame.id) if frame else None
        scene_assignments = assignments[scene.id]
        transcript_text = " ".join(segment.text for segment, _score in scene_assignments).strip()
        average_score = (
            sum(score for _, score in scene_assignments) / len(scene_assignments)
            if scene_assignments
            else 0.0
        )

        blocks.append(
            AlignmentBlock(
                id=f"block-{index}",
                start_sec=scene.start_sec,
                end_sec=scene.end_sec,
                transcript_segment_ids=[segment.id for segment, _score in scene_assignments],
                transcript_text=transcript_text,
                scene_id=scene.id,
                frame_id=frame.id if frame else None,
                title_hint=_summarize_ocr_text(ocr_result.text) if ocr_result else None,
                scores={"hybrid": round(average_score, 4)},
            )
        )

    return blocks


def _is_informative_ocr(result: OcrResult) -> bool:
    return bool(result.text.strip()) and (result.confidence or 0.0) >= 0.3


def _time_score(segment: TranscriptSegment, scene: Scene) -> float:
    midpoint = (segment.start_sec + segment.end_sec) / 2
    scene_midpoint = (scene.start_sec + scene.end_sec) / 2
    scene_duration = max(scene.end_sec - scene.start_sec, 0.001)
    distance = abs(midpoint - scene_midpoint)
    return max(0.0, 1.0 - min(distance / scene_duration, 1.0))


def _text_score(segment_text: str, ocr_text: str) -> float:
    normalized_segment = " ".join(TOKEN_PATTERN.findall(segment_text.lower()))
    normalized_ocr = " ".join(TOKEN_PATTERN.findall(ocr_text.lower()))
    if not normalized_segment or not normalized_ocr:
        return 0.0
    return fuzz.token_set_ratio(normalized_segment, normalized_ocr) / 100


def _summarize_ocr_text(text: str) -> str | None:
    words = TOKEN_PATTERN.findall(text.strip())
    if not words:
        return None
    return " ".join(words[:6])
