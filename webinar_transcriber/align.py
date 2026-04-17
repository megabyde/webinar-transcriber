"""Alignment helpers for transcript segments and video scenes."""

from dataclasses import replace

from webinar_transcriber.models import AlignmentBlock, Scene, SlideFrame, TranscriptSegment


def align_by_time(
    transcript_segments: list[TranscriptSegment],
    scenes: list[Scene],
    slide_frames: list[SlideFrame],
) -> list[AlignmentBlock]:
    """Assign transcript segments to scenes using midpoint inclusion.

    Returns:
        list[AlignmentBlock]: The scene-aligned transcript blocks.
    """
    frame_by_scene = {frame.scene_id: frame for frame in slide_frames}
    blocks: list[AlignmentBlock] = []
    assigned_segment_ids: set[str] = set()

    for index, scene in enumerate(scenes, start=1):
        scene_segments = [
            segment
            for segment in transcript_segments
            if scene.start_sec <= segment.midpoint < scene.end_sec
        ]
        assigned_segment_ids.update(segment.id for segment in scene_segments)
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

    if blocks:
        for segment in transcript_segments:
            if segment.id in assigned_segment_ids:
                continue
            target_index = next(
                (
                    index
                    for index in reversed(range(len(blocks)))
                    if blocks[index].start_sec <= segment.midpoint
                ),
                0,
            )
            block = blocks[target_index]
            block_segments = sorted(
                [
                    *[
                        segment
                        for segment in transcript_segments
                        if segment.id in block.transcript_segment_ids
                    ],
                    segment,
                ],
                key=lambda item: (item.start_sec, item.end_sec, item.id),
            )
            blocks[target_index] = replace(
                block,
                transcript_segment_ids=[segment.id for segment in block_segments],
                transcript_text=" ".join(
                    segment.text for segment in block_segments if segment.text
                ).strip(),
            )

    return blocks
