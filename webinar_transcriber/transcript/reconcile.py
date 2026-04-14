"""Reconcile decoded windows into one monotonic transcript."""

from __future__ import annotations

from dataclasses import dataclass

from webinar_transcriber.models import DecodedWindow, TranscriptionResult, TranscriptSegment


@dataclass(frozen=True)
class ReconciliationStats:
    """Observed cleanup performed during transcript reconciliation."""

    duplicate_segments_dropped: int = 0
    boundary_fixes: int = 0


def reconcile_decoded_windows(
    decoded_windows: list[DecodedWindow],
) -> tuple[TranscriptionResult, ReconciliationStats]:
    """Merge decoded windows while keeping segment boundaries monotonic."""
    ordered_windows = sorted(decoded_windows, key=lambda item: item.window)
    detected_language = next(
        (window.language for window in ordered_windows if window.language), None
    )

    reconciled_segments: list[TranscriptSegment] = []
    boundary_fixes = 0

    for decoded_window in ordered_windows:
        for segment in decoded_window.segments:
            text = segment.text.strip()
            if not text:
                continue

            start_sec = max(0.0, segment.start_sec)
            if reconciled_segments and start_sec < reconciled_segments[-1].end_sec:
                start_sec = reconciled_segments[-1].end_sec
                boundary_fixes += 1

            end_sec = segment.end_sec
            if end_sec < start_sec:
                end_sec = start_sec
                boundary_fixes += 1

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

    return (
        TranscriptionResult(detected_language=detected_language, segments=reconciled_segments),
        ReconciliationStats(boundary_fixes=boundary_fixes),
    )
