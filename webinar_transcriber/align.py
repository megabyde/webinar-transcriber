"""Alignment helpers for transcript segments and video scenes."""

from webinar_transcriber.labels import count_label
from webinar_transcriber.models import AlignmentBlock, Scene, SlideFrame, TranscriptSegment


def align_by_time(
    transcript_segments: list[TranscriptSegment],
    scenes: list[Scene],
    slide_frames: list[SlideFrame],
    *,
    warnings: list[str] | None = None,
) -> list[AlignmentBlock]:
    """Assign transcript segments to scenes using midpoint inclusion."""
    frame_by_scene = {frame.scene_id: frame for frame in slide_frames}
    blocks: list[AlignmentBlock] = []
    segments_by_block: list[list[TranscriptSegment]] = []
    assigned_segment_ids: set[str] = set()

    for index, scene in enumerate(scenes, start=1):
        scene_segments = [
            segment
            for segment in transcript_segments
            if scene.start_sec <= segment.midpoint < scene.end_sec
        ]
        assigned_segment_ids.update(segment.id for segment in scene_segments)
        segments_by_block.append(list(scene_segments))
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

    orphan_segments = [
        segment for segment in transcript_segments if segment.id not in assigned_segment_ids
    ]
    if orphan_segments and blocks:
        for segment in orphan_segments:
            nearest_block_index = min(
                range(len(blocks)),
                key=lambda index: _distance_to_block(segment.midpoint, blocks[index]),
            )
            segments_by_block[nearest_block_index].append(segment)

        for block, block_segments in zip(blocks, segments_by_block, strict=False):
            ordered_segments = sorted(block_segments)
            block.transcript_segment_ids = [segment.id for segment in ordered_segments]
            block.transcript_text = " ".join(
                segment.text for segment in ordered_segments if segment.text
            ).strip()

        if warnings is not None:
            warnings.append(_orphan_alignment_warning(orphan_count=len(orphan_segments)))

    return blocks


def _distance_to_block(midpoint_sec: float, block: AlignmentBlock) -> float:
    if midpoint_sec < block.start_sec:
        return block.start_sec - midpoint_sec
    if midpoint_sec > block.end_sec:
        return midpoint_sec - block.end_sec
    return 0.0


def _orphan_alignment_warning(*, orphan_count: int) -> str:
    return (
        f"Aligned {count_label(orphan_count, 'transcript segment')} to the nearest "
        "scene blocks because their midpoints fell outside all scene ranges."
    )
