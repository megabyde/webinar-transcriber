"""Title, summary, and action-item scoring helpers for report structuring."""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING

from .constants import (
    ACTION_ITEM_LIMIT,
    ACTION_ITEM_PATTERNS,
    SUMMARY_ITEM_LIMIT,
    SUMMARY_NOISE_PATTERN,
    TITLE_FILLER_WORDS,
    TITLE_WORD_LIMIT,
)

if TYPE_CHECKING:
    from webinar_transcriber.models import TranscriptSegment


def _build_summary(segments: list[TranscriptSegment]) -> list[str]:
    candidates: list[tuple[float, int, str]] = []

    for index, segment in enumerate(segments):
        text = segment.text.strip()
        if not text:
            continue

        score = _summary_score(segment)
        candidates.append((score, index, text))

    selected: list[tuple[int, str]] = []
    seen_keys: set[str] = set()
    for score, index, text in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if score <= 0:
            continue
        key = _segment_key(text)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append((index, text))
        if len(selected) == SUMMARY_ITEM_LIMIT:
            break

    if not selected:
        return _fallback_summary(segments)

    return [text for _, text in sorted(selected, key=lambda item: item[0])]


def _extract_action_items(segments: list[TranscriptSegment]) -> list[str]:
    candidates: list[tuple[float, int, str]] = []
    seen_keys: set[str] = set()

    for index, segment in enumerate(segments):
        text = segment.text.strip()
        if not text or not _has_action_item_cue(text):
            continue
        score = _action_item_score(segment)
        if score <= 0:
            continue
        candidates.append((score, index, text))

    selected: list[tuple[int, str]] = []
    for _score, index, text in sorted(candidates, key=lambda item: (-item[0], item[1])):
        key = _segment_key(text)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append((index, text))
        if len(selected) == ACTION_ITEM_LIMIT:
            break

    return [text for _, text in sorted(selected, key=lambda item: item[0])]


def _title_from_text(text: str, *, fallback: str) -> str:
    cleaned = text.strip().rstrip(".")
    if not cleaned:
        return fallback

    words = cleaned.split()
    return " ".join(words[:TITLE_WORD_LIMIT]) if len(words) > TITLE_WORD_LIMIT else cleaned


def _audio_title_from_segments(segments: list[TranscriptSegment], *, fallback: str) -> str:
    best_segment: TranscriptSegment | None = None
    best_score = float("-inf")

    for segment in segments:
        score = _audio_title_score(segment)
        if score > best_score:
            best_segment = segment
            best_score = score

    if best_segment is None:
        return fallback

    title_words = _title_words(best_segment.text)
    title = _title_from_words(title_words)
    return title or fallback


def _audio_title_score(segment: TranscriptSegment) -> float:
    words = _title_words(segment.text)
    if len(words) < 4:
        return -1.0

    informative_words = sum(
        1 for word in words[:12] if word not in TITLE_FILLER_WORDS and len(word) > 2
    )
    if informative_words < 3:
        return -1.0

    unique_ratio = len(set(words[:12])) / min(len(words), 12)
    score = float(informative_words)
    if unique_ratio < 0.6:
        score -= 3.0
    return score


def _summary_score(segment: TranscriptSegment) -> float:
    text = segment.text.strip()
    words = _title_words(text)
    word_count = len(words)
    if word_count < 4:
        return -2.0

    if SUMMARY_NOISE_PATTERN.search(text):
        return -2.0

    score = -1.0 if segment.start_sec < 60.0 else 0.0
    informative_words = sum(1 for word in words[:14] if len(word) > 2)
    duration = max(0.0, segment.end_sec - segment.start_sec)
    score += informative_words + min(duration, 15.0) / 5.0
    unique_ratio = len(set(words[:14])) / min(len(words), 14)
    if unique_ratio < 0.6:
        score -= 2.0
    return score


def _action_item_score(segment: TranscriptSegment) -> float:
    text = segment.text.strip()
    words = _title_words(text)
    if len(words) < 2:
        return -1.0

    if SUMMARY_NOISE_PATTERN.search(text):
        return -1.0

    score = 1.0
    if segment.start_sec >= 60.0:
        score += 1.0
    return score


def _title_words(text: str) -> list[str]:
    words = re.findall(r"[\w'-]+", text.lower())
    start_index = 0
    while start_index < len(words) and words[start_index] in TITLE_FILLER_WORDS:
        start_index += 1
    return words[start_index:]


def _title_from_words(words: list[str]) -> str:
    if not words:
        return ""

    title = " ".join(words[:TITLE_WORD_LIMIT])
    return title[:1].upper() + title[1:]


def _segment_key(text: str) -> str:
    words = _title_words(text)
    if not words:
        return text.strip().lower()
    return " ".join(words[:8])


def _fallback_summary(segments: list[TranscriptSegment]) -> list[str]:
    summary: list[str] = []

    for segment in segments:
        text = segment.text.strip()
        if text and text not in summary:
            summary.append(text)
        if len(summary) == SUMMARY_ITEM_LIMIT:
            break

    return summary


def _has_action_item_cue(text: str) -> bool:
    return any(pattern.search(text) for pattern in ACTION_ITEM_PATTERNS)


def _derive_title(source_path: str) -> str:
    stem = PureWindowsPath(source_path).stem if "\\" in source_path else Path(source_path).stem
    return stem.replace("-", " ").replace("_", " ").strip().title() or "Transcription Report"
