"""Reconcile decoded windows into one monotonic transcript."""

from __future__ import annotations

from webinar_transcriber.models import DecodedWindow, TranscriptionResult, TranscriptSegment


def reconcile_decoded_windows(
    decoded_windows: list[DecodedWindow],
) -> TranscriptionResult:
    """Merge decoded windows while keeping segment boundaries monotonic.

    Returns:
        TranscriptionResult: The reconciled transcript.
    """
    ordered_windows = sorted(
        decoded_windows,
        key=lambda item: (
            item.window.start_sec,
            item.window.end_sec,
            item.window.region_index,
            item.window.window_id,
        ),
    )
    detected_language = next(
        (window.language for window in ordered_windows if window.language), None
    )

    reconciled_segments: list[TranscriptSegment] = []
    for decoded_window in ordered_windows:
        for segment in decoded_window.segments:
            text = segment.text.strip()
            if not text:
                continue

            start_sec = max(0.0, segment.start_sec)
            if reconciled_segments and start_sec < reconciled_segments[-1].end_sec:
                start_sec = reconciled_segments[-1].end_sec

            end_sec = max(segment.end_sec, start_sec)

            if end_sec <= start_sec:
                continue

            reconciled_segments.append(
                TranscriptSegment(
                    id=f"segment-{len(reconciled_segments) + 1}",
                    text=text,
                    start_sec=start_sec,
                    end_sec=end_sec,
                )
            )

    return TranscriptionResult(detected_language=detected_language, segments=reconciled_segments)
