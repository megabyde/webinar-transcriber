"""Deterministic report structuring helpers."""

import re

from webinar_transcriber.models import (
    AlignmentBlock,
    MediaAsset,
    ReportDocument,
    ReportSection,
    TranscriptionResult,
    TranscriptSegment,
)

ACTION_ITEM_PATTERN = re.compile(
    r"\b(action item|follow up|follow-up|next step|todo|we should|please)\b",
    re.IGNORECASE,
)
AUDIO_SECTION_BREAK_GAP_SEC = 8.0
TARGET_AUDIO_SECTION_DURATION_SEC = 300.0
MIN_AUDIO_SECTION_DURATION_SEC = 120.0
MAX_AUDIO_SECTION_CHARS = 3600
TITLE_WORD_LIMIT = 6
TITLE_FILLER_WORDS = {
    "actually",
    "basically",
    "just",
    "like",
    "okay",
    "right",
    "so",
    "then",
    "well",
    "you",
    "бы",
    "в",
    "вот",
    "да",
    "други",
    "как",
    "короче",
    "ладно",
    "ну",
    "парни",
    "просто",
    "ребят",
    "скажем",
    "собственно",
    "соответственно",
    "значит",
    "так",
    "типа",
    "то",
    "хорошо",
    "что",
    "это",
}


def build_report(
    media_asset: MediaAsset,
    transcription: TranscriptionResult,
    *,
    alignment_blocks: list[AlignmentBlock] | None = None,
    warnings: list[str] | None = None,
) -> ReportDocument:
    """Build a report document from media metadata and transcript segments."""
    sections = (
        _build_sections_from_blocks(alignment_blocks)
        if alignment_blocks is not None
        else _build_audio_sections(transcription.segments)
    )
    summary = _build_summary(transcription.segments)
    action_items = _extract_action_items(transcription.segments)

    return ReportDocument(
        title=_derive_title(media_asset.path),
        source_file=media_asset.path,
        media_type=media_asset.media_type,
        detected_language=transcription.detected_language,
        summary=summary,
        action_items=action_items,
        sections=sections,
        warnings=warnings or [],
    )


def _build_sections_from_blocks(blocks: list[AlignmentBlock]) -> list[ReportSection]:
    sections: list[ReportSection] = []

    for index, block in enumerate(blocks, start=1):
        title = _title_from_text(block.transcript_text, fallback=f"Slide {index}")
        if block.title_hint:
            title = _title_from_text(block.title_hint, fallback=title)
        sections.append(
            ReportSection(
                id=f"section-{index}",
                title=title,
                start_sec=block.start_sec,
                end_sec=block.end_sec,
                transcript_text=block.transcript_text,
                frame_id=block.frame_id,
            )
        )

    return sections


def _build_audio_sections(segments: list[TranscriptSegment]) -> list[ReportSection]:
    meaningful_segments = [segment for segment in segments if segment.text.strip()]
    if not meaningful_segments:
        return []

    sections: list[ReportSection] = []
    current_segments: list[TranscriptSegment] = []

    for segment in meaningful_segments:
        if _should_start_new_audio_section(current_segments, segment):
            sections.append(_audio_section_from_segments(current_segments, len(sections) + 1))
            current_segments = []

        current_segments.append(segment)

    if current_segments:
        sections.append(_audio_section_from_segments(current_segments, len(sections) + 1))

    return sections


def _should_start_new_audio_section(
    current_segments: list[TranscriptSegment], next_segment: TranscriptSegment
) -> bool:
    if not current_segments:
        return False

    current_start = current_segments[0].start_sec
    current_end = current_segments[-1].end_sec
    current_duration = current_end - current_start
    gap_duration = max(0.0, next_segment.start_sec - current_end)
    next_duration = max(0.0, next_segment.end_sec - current_start)
    current_chars = sum(len(segment.text.strip()) for segment in current_segments)

    if gap_duration >= AUDIO_SECTION_BREAK_GAP_SEC:
        return True

    if (
        current_duration >= MIN_AUDIO_SECTION_DURATION_SEC
        and next_duration > TARGET_AUDIO_SECTION_DURATION_SEC
    ):
        return True

    return (
        current_duration >= MIN_AUDIO_SECTION_DURATION_SEC
        and current_chars >= MAX_AUDIO_SECTION_CHARS
    )


def _audio_section_from_segments(
    segments: list[TranscriptSegment], section_index: int
) -> ReportSection:
    transcript_text = "\n\n".join(
        segment.text.strip() for segment in segments if segment.text.strip()
    )
    title = _audio_title_from_segments(segments, fallback=f"Section {section_index}")

    return ReportSection(
        id=f"section-{section_index}",
        title=title,
        start_sec=segments[0].start_sec,
        end_sec=segments[-1].end_sec,
        transcript_text=transcript_text,
    )


def _build_summary(segments: list[TranscriptSegment]) -> list[str]:
    summary: list[str] = []

    for segment in segments:
        text = segment.text.strip()
        if text and text not in summary:
            summary.append(text)
        if len(summary) == 3:
            break

    return summary


def _extract_action_items(segments: list[TranscriptSegment]) -> list[str]:
    return [
        segment.text.strip() for segment in segments if ACTION_ITEM_PATTERN.search(segment.text)
    ]


def _title_from_text(text: str, *, fallback: str) -> str:
    cleaned = text.strip().rstrip(".")
    if not cleaned:
        return fallback

    words = cleaned.split()
    return " ".join(words[:TITLE_WORD_LIMIT]) if len(words) > TITLE_WORD_LIMIT else cleaned


def _audio_title_from_segments(segments: list[TranscriptSegment], *, fallback: str) -> str:
    best_segment: TranscriptSegment | None = None
    best_score = float("-inf")

    for index, segment in enumerate(segments):
        adjusted_score = _audio_title_score(segment) - min(index, 5) * 0.15
        if adjusted_score > best_score:
            best_segment = segment
            best_score = adjusted_score

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
    punctuation_bonus = 1.0 if any(char in segment.text for char in ".?!,:;") else 0.0
    unique_ratio = len(set(words[:12])) / min(len(words), 12)
    repetition_penalty = 0.0
    if unique_ratio < 0.4:
        repetition_penalty = 4.0
    elif unique_ratio < 0.6:
        repetition_penalty = 2.0
    return informative_words + punctuation_bonus + min(len(words), 12) / 20.0 - repetition_penalty


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


def _derive_title(source_path: str) -> str:
    stem = source_path.rsplit("/", maxsplit=1)[-1].rsplit(".", maxsplit=1)[0]
    return stem.replace("-", " ").replace("_", " ").strip().title() or "Transcription Report"
