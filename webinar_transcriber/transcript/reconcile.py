"""Reconcile overlapping window transcripts into a single monotonic transcript.

The overlap solver stays in Python because it is policy: we want deterministic local heuristics
that can evolve independently of the whisper.cpp binding. The binding only returns per-window
segments; this module decides which boundary words survive when adjacent windows disagree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from webinar_transcriber.models import DecodedWindow, TranscriptionResult, TranscriptSegment

_NON_WORD_PATTERN = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True)
class ReconciliationStats:
    """Observed cleanup performed during overlap reconciliation."""

    duplicate_segments_dropped: int = 0
    boundary_fixes: int = 0


@dataclass(frozen=True)
class _TokenPiece:
    text: str
    normalized: str
    start_sec: float
    end_sec: float
    window_start_sec: float
    window_end_sec: float
    window_order: int
    segment_order: int
    token_order: int
    source_segment_id: str

    @property
    def center_sec(self) -> float:
        return self.start_sec + ((self.end_sec - self.start_sec) / 2.0)


def reconcile_decoded_windows(
    decoded_windows: list[DecodedWindow],
) -> tuple[TranscriptionResult, ReconciliationStats]:
    """Merge decoded windows while resolving overlap by normalized-token alignment."""
    detected_language: str | None = None
    merged_tokens: list[_TokenPiece] = []
    duplicate_segments_dropped = 0
    boundary_fixes = 0

    ordered_windows = sorted(decoded_windows, key=lambda item: item.window)
    for window_order, decoded_window in enumerate(ordered_windows):
        detected_language = detected_language or decoded_window.language

        current_tokens = _window_tokens(decoded_window, window_order=window_order)
        if not current_tokens:
            continue

        if merged_tokens:
            drop_previous, drop_current = _duplicate_token_indices(merged_tokens, current_tokens)
            if drop_previous or drop_current:
                boundary_fixes += 1
            previous_segment_ids = {
                merged_tokens[index].source_segment_id for index in drop_previous
            }
            current_segment_ids = {
                current_tokens[index].source_segment_id for index in drop_current
            }
            duplicate_segments_dropped += _fully_dropped_segment_count(
                merged_tokens,
                drop_previous,
                previous_segment_ids,
            )
            duplicate_segments_dropped += _fully_dropped_segment_count(
                current_tokens,
                drop_current,
                current_segment_ids,
            )
            merged_tokens = [
                token for index, token in enumerate(merged_tokens) if index not in drop_previous
            ]
            current_tokens = [
                token for index, token in enumerate(current_tokens) if index not in drop_current
            ]

        merged_tokens = sorted(
            [*merged_tokens, *current_tokens],
            key=lambda token: (
                token.start_sec,
                token.end_sec,
                token.window_order,
                token.segment_order,
                token.token_order,
            ),
        )

    segments, extra_boundary_fixes = _segments_from_tokens(merged_tokens)
    return (
        TranscriptionResult(detected_language=detected_language, segments=segments),
        ReconciliationStats(
            duplicate_segments_dropped=duplicate_segments_dropped,
            boundary_fixes=boundary_fixes + extra_boundary_fixes,
        ),
    )


def _window_tokens(decoded_window: DecodedWindow, *, window_order: int) -> list[_TokenPiece]:
    tokens: list[_TokenPiece] = []
    for segment_order, segment in enumerate(decoded_window.segments):
        split_tokens = segment.text.split()
        if not split_tokens:
            continue

        # Token timing is linearly interpolated across whitespace tokens, which is cheap and
        # deterministic but only an approximation until whisper.cpp token timestamps are wired in.
        duration_sec = max(0.0, segment.end_sec - segment.start_sec)
        token_duration_sec = duration_sec / len(split_tokens) if split_tokens else 0.0
        for token_order, token_text in enumerate(split_tokens):
            start_sec = segment.start_sec + (token_duration_sec * token_order)
            end_sec = (
                segment.end_sec
                if token_order == len(split_tokens) - 1
                else segment.start_sec + (token_duration_sec * (token_order + 1))
            )
            tokens.append(
                _TokenPiece(
                    text=token_text,
                    normalized=_normalize_token(token_text),
                    start_sec=max(0.0, start_sec),
                    end_sec=max(start_sec, end_sec),
                    window_start_sec=decoded_window.window.start_sec,
                    window_end_sec=decoded_window.window.end_sec,
                    window_order=window_order,
                    segment_order=segment_order,
                    token_order=token_order,
                    source_segment_id=segment.id,
                )
            )
    return tokens


def _duplicate_token_indices(
    previous_tokens: list[_TokenPiece],
    current_tokens: list[_TokenPiece],
) -> tuple[set[int], set[int]]:
    if not previous_tokens or not current_tokens:
        return set(), set()

    current_window_start = current_tokens[0].start_sec
    current_window_end = current_tokens[-1].end_sec
    overlap_region_start = current_window_start
    overlap_region_end = min(previous_tokens[-1].end_sec, current_window_end)
    if overlap_region_end <= overlap_region_start:
        return set(), set()

    previous_overlap = [
        (index, token)
        for index, token in enumerate(previous_tokens)
        if token.end_sec >= overlap_region_start and token.start_sec < overlap_region_end
    ]
    current_overlap = [
        (index, token)
        for index, token in enumerate(current_tokens)
        if token.end_sec >= overlap_region_start and token.start_sec < overlap_region_end
    ]
    matches = _align_overlap(previous_overlap, current_overlap)
    if not matches:
        if _mean_edge_distance(previous_overlap) >= _mean_edge_distance(current_overlap):
            return set(), {index for index, _ in current_overlap}
        return {index for index, _ in previous_overlap}, set()

    midpoint_sec = overlap_region_start + ((overlap_region_end - overlap_region_start) / 2.0)
    drop_previous: set[int] = set()
    drop_current: set[int] = set()
    for match_group in _group_match_runs(matches):
        current_centers = [
            current_tokens[current_index].center_sec for _, current_index in match_group
        ]
        if (sum(current_centers) / len(current_centers)) < midpoint_sec:
            drop_current.update(current_index for _, current_index in match_group)
        else:
            drop_previous.update(previous_index for previous_index, _ in match_group)
    return drop_previous, drop_current


def _align_overlap(
    previous_overlap: list[tuple[int, _TokenPiece]],
    current_overlap: list[tuple[int, _TokenPiece]],
) -> list[tuple[int, int]]:
    # This uses an O(n^2) LCS table and assumes overlap regions stay short.
    previous_filtered = [(index, token) for index, token in previous_overlap if token.normalized]
    current_filtered = [(index, token) for index, token in current_overlap if token.normalized]
    if not previous_filtered or not current_filtered:
        return []

    rows = len(previous_filtered)
    cols = len(current_filtered)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]

    for row in range(rows):
        for col in range(cols):
            if previous_filtered[row][1].normalized == current_filtered[col][1].normalized:
                table[row + 1][col + 1] = table[row][col] + 1
            else:
                table[row + 1][col + 1] = max(table[row][col + 1], table[row + 1][col])

    matches: list[tuple[int, int]] = []
    row = rows
    col = cols
    while row > 0 and col > 0:
        previous_index, previous_token = previous_filtered[row - 1]
        current_index, current_token = current_filtered[col - 1]
        if previous_token.normalized == current_token.normalized:
            matches.append((previous_index, current_index))
            row -= 1
            col -= 1
        elif table[row - 1][col] >= table[row][col - 1]:
            row -= 1
        else:
            col -= 1
    matches.reverse()
    return matches


def _group_match_runs(matches: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    if not matches:
        return []

    groups: list[list[tuple[int, int]]] = [[matches[0]]]
    for previous_index, current_index in matches[1:]:
        last_previous, last_current = groups[-1][-1]
        if previous_index == last_previous + 1 and current_index == last_current + 1:
            groups[-1].append((previous_index, current_index))
            continue
        groups.append([(previous_index, current_index)])
    return groups


def _fully_dropped_segment_count(
    tokens: list[_TokenPiece],
    dropped_indices: set[int],
    candidate_segment_ids: set[str],
) -> int:
    fully_dropped = 0
    for segment_id in candidate_segment_ids:
        segment_indices = {
            index for index, token in enumerate(tokens) if token.source_segment_id == segment_id
        }
        if segment_indices and segment_indices.issubset(dropped_indices):
            fully_dropped += 1
    return fully_dropped


def _mean_edge_distance(overlap: list[tuple[int, _TokenPiece]]) -> float:
    if not overlap:
        return 0.0
    distances = [
        min(
            token.center_sec - token.window_start_sec,
            token.window_end_sec - token.center_sec,
        )
        for _, token in overlap
    ]
    return sum(distances) / len(distances)


def _segments_from_tokens(
    tokens: list[_TokenPiece],
) -> tuple[list[TranscriptSegment], int]:
    if not tokens:
        return [], 0

    # Reconciliation preserves source-window seams here; normalize_transcription downstream is
    # responsible for merging those fragments back into cleaner utterance boundaries.
    segments: list[TranscriptSegment] = []
    boundary_fixes = 0
    current_group: list[_TokenPiece] = [tokens[0]]

    def flush_group(group: list[_TokenPiece], segment_index: int) -> None:
        nonlocal boundary_fixes
        text = " ".join(token.text for token in group).strip()
        if not text:
            return
        start_sec = max(0.0, group[0].start_sec)
        end_sec = max(start_sec, group[-1].end_sec)
        if segments and start_sec < segments[-1].end_sec:
            start_sec = segments[-1].end_sec
            boundary_fixes += 1
        if end_sec < start_sec:
            end_sec = start_sec
            boundary_fixes += 1
        if end_sec == start_sec:
            return
        segments.append(
            TranscriptSegment(
                id=f"segment-{segment_index}",
                text=text,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )

    segment_index = 1
    for token in tokens[1:]:
        previous = current_group[-1]
        same_source = (
            token.source_segment_id == previous.source_segment_id
            and token.window_order == previous.window_order
            and token.segment_order == previous.segment_order
            and token.token_order == previous.token_order + 1
        )
        if same_source:
            current_group.append(token)
            continue
        flush_group(current_group, segment_index)
        segment_index += 1
        current_group = [token]
    flush_group(current_group, segment_index)
    return segments, boundary_fixes


def _normalize_token(text: str) -> str:
    return _NON_WORD_PATTERN.sub("", text.casefold())
