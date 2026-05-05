"""Reconcile decoded windows into one monotonic transcript."""

from __future__ import annotations

import re

from webinar_transcriber.models import DecodedWindow, TranscriptionResult, TranscriptSegment

_WORD_RE = re.compile(r"\w+")
_MIN_DUPLICATE_OVERLAP_WORDS = 3
_MAX_DUPLICATE_OVERLAP_WORDS = 80


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
            _window_id_sort_value(item.window.window_id),
        ),
    )
    detected_language = next(
        (window.language for window in ordered_windows if window.language), None
    )

    reconciled_segments: list[TranscriptSegment] = []
    covered_until_sec = 0.0
    for decoded_window in ordered_windows:
        for segment in decoded_window.segments:
            text = segment.text.strip()
            if not text:
                continue
            if reconciled_segments and segment.start_sec < covered_until_sec:
                text = _trim_duplicate_prefix(
                    _recent_transcript_text(reconciled_segments), text
                ).strip()
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
        covered_until_sec = max(covered_until_sec, decoded_window.window.end_sec)

    return TranscriptionResult(detected_language=detected_language, segments=reconciled_segments)


def _window_id_sort_value(window_id: str) -> tuple[str, int]:
    prefix, sep, suffix = window_id.rpartition("-")
    if sep and suffix.isdecimal():
        return prefix, int(suffix)
    return window_id, -1


def _recent_transcript_text(segments: list[TranscriptSegment]) -> str:
    return " ".join(segment.text for segment in segments[-8:])


def _trim_duplicate_prefix(previous_text: str, text: str) -> str:
    previous_words = _word_spans(previous_text)
    current_words = _word_spans(text)
    max_overlap = min(len(previous_words), len(current_words), _MAX_DUPLICATE_OVERLAP_WORDS)
    for overlap_word_count in range(max_overlap, _MIN_DUPLICATE_OVERLAP_WORDS - 1, -1):
        previous_suffix = previous_words[-overlap_word_count:]
        current_prefix = current_words[:overlap_word_count]
        if [word for word, _start, _end in previous_suffix] != [
            word for word, _start, _end in current_prefix
        ]:
            continue
        if overlap_word_count == len(current_words):
            return ""
        trim_end = current_prefix[-1][2]
        return text[trim_end:].lstrip(" \t\r\n,.;:!?-")
    return text


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group().casefold(), match.start(), match.end()) for match in _WORD_RE.finditer(text)
    ]
