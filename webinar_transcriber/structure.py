"""Deterministic report structuring helpers."""

import re
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path, PureWindowsPath

from webinar_transcriber.models import (
    AlignmentBlock,
    MediaAsset,
    ReportDocument,
    ReportSection,
    TranscriptionResult,
    TranscriptSegment,
)
from webinar_transcriber.transcript_processing import STRONG_SENTENCE_END_RE


def _compile_case_insensitive_patterns(*patterns: str) -> tuple[re.Pattern[str], ...]:
    """Compile a set of regex patterns with consistent case-insensitive flags."""
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


EN_ACTION_PATTERN = (
    r"\b(?:action item|follow[ -]?up|next step|todo|"
    r"please (?:follow up|send|share|review|check|update|remember|try))\b"
)
EN_REMINDER_PATTERN = r"\b(?:remember to|make sure to)\b"
EN_HOMEWORK_PATTERN = r"\bhome ?work\b"
RU_HOMEWORK_PATTERN = (
    r"\bдомашн(?:ее|е) "  # noqa: RUF001
    r"задание\b"
)
RU_REMINDER_PATTERN = r"\bне забудьте\b"  # noqa: RUF001
RU_ACTION_PATTERN = (
    r"\bпожалуйста[, ]+(?:"  # noqa: RUF001
    r"пришлите|"
    r"напишите|"
    r"проверьте|"
    r"сделайте|"
    r"отправьте|"
    r"подготовьте)\b"
)

ACTION_ITEM_PATTERNS = _compile_case_insensitive_patterns(
    EN_ACTION_PATTERN,
    EN_REMINDER_PATTERN,
    EN_HOMEWORK_PATTERN,
    RU_ACTION_PATTERN,
    RU_REMINDER_PATTERN,
    RU_HOMEWORK_PATTERN,
)
AUDIO_SECTION_BREAK_GAP_SEC = 8.0
TARGET_AUDIO_SECTION_DURATION_SEC = 300.0
MIN_AUDIO_SECTION_DURATION_SEC = 120.0
MAX_AUDIO_SECTION_CHARS = 3600
TITLE_WORD_LIMIT = 6
SUMMARY_ITEM_LIMIT = 3
ACTION_ITEM_LIMIT = 5
INTERLUDE_MIN_WORDS = 18
MIN_INTERLUDE_DURATION_SEC = 30.0
INTERLUDE_LOW_PUNCTUATION_DENSITY = 0.06
INTERLUDE_LOW_UNIQUE_RATIO = 0.5
INTERLUDE_REPEATED_BIGRAM_RATIO = 0.12
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
SUMMARY_NOISE_PATTERN = re.compile(
    r"\b("
    r"звук|слышно|микрофон|всем привет|добрый вечер|чат|чате|групп[аы]|"
    r"sound|audio|mic|microphone|hello everyone|good evening|chat|group"
    r")\b",
    re.IGNORECASE,
)
INTERLUDE_MARKER_PATTERN = re.compile(
    r"(?:"
    r"субтитры сделал|музыкальн|припев|куплет|"
    r"lyrics?|instrumental|chorus|verse|interlude"
    r")",
    re.IGNORECASE,
)
INTERLUDE_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
RUSSIAN_VOWELS = frozenset("аеёиоуыэюя")
LATIN_VOWELS = frozenset("aeiouy")


def build_report(
    media_asset: MediaAsset,
    transcription: TranscriptionResult,
    *,
    alignment_blocks: list[AlignmentBlock] | None = None,
    warnings: list[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ReportDocument:
    """Build a report document from media metadata and transcript segments."""
    interlude_candidate_ranges = _detect_interlude_ranges(transcription.segments)
    interlude_ranges = _renderable_interlude_ranges(interlude_candidate_ranges)
    sections = (
        _build_sections_from_blocks(
            alignment_blocks,
            transcript_segments=transcription.segments,
            interlude_ranges=interlude_ranges,
            progress_callback=progress_callback,
        )
        if alignment_blocks is not None
        else _build_audio_sections(
            transcription.segments,
            interlude_ranges=interlude_ranges,
            progress_callback=progress_callback,
        )
    )
    sections = _render_interlude_sections(
        sections,
        detected_language=transcription.detected_language,
    )
    report_segments = _segments_excluding_interludes(transcription.segments, sections)
    summary = _build_summary(report_segments)
    action_items = _extract_action_items(report_segments)

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


def _build_sections_from_blocks(
    blocks: list[AlignmentBlock],
    *,
    transcript_segments: list[TranscriptSegment],
    interlude_ranges: list[tuple[float, float]],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[ReportSection]:
    segment_by_id = {segment.id: segment for segment in transcript_segments}
    sections: list[ReportSection] = []

    for index, block in enumerate(blocks, start=1):
        block_segments = [
            segment_by_id[segment_id]
            for segment_id in block.transcript_segment_ids
            if segment_id in segment_by_id and segment_by_id[segment_id].text.strip()
        ]
        sections.extend(
            _sections_from_block(
                block,
                block_segments=block_segments,
                interlude_ranges=interlude_ranges,
                next_section_index=len(sections) + 1,
            )
        )
        if progress_callback is not None:
            progress_callback(index, len(sections))

    return sections


def _build_audio_sections(
    segments: list[TranscriptSegment],
    *,
    interlude_ranges: list[tuple[float, float]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[ReportSection]:
    resolved_interlude_ranges = interlude_ranges or []
    meaningful_segments = [seg for seg in segments if seg.text.strip()]
    if not meaningful_segments:
        return []

    sections: list[ReportSection] = []
    current_speech_segments: list[TranscriptSegment] = []
    current_interlude_segments: list[TranscriptSegment] = []

    for index, segment in enumerate(meaningful_segments, start=1):
        if _overlaps_interlude_ranges(segment, resolved_interlude_ranges):
            if current_speech_segments:
                sections.append(
                    _audio_section_from_segments(
                        current_speech_segments,
                        len(sections) + 1,
                    )
                )
                current_speech_segments = []
            current_interlude_segments.append(segment)
        else:
            if current_interlude_segments:
                sections.append(
                    _interlude_section_from_segments(
                        current_interlude_segments,
                        len(sections) + 1,
                    )
                )
                current_interlude_segments = []
            if _should_start_new_audio_section(current_speech_segments, segment):
                sections.append(
                    _audio_section_from_segments(
                        current_speech_segments,
                        len(sections) + 1,
                    )
                )
                current_speech_segments = []
            current_speech_segments.append(segment)
        if progress_callback is not None:
            progress_callback(index, len(sections) + 1)

    if current_speech_segments:
        sections.append(_audio_section_from_segments(current_speech_segments, len(sections) + 1))
    if current_interlude_segments:
        sections.append(
            _interlude_section_from_segments(
                current_interlude_segments,
                len(sections) + 1,
            )
        )

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
    ends_on_sentence_boundary = bool(
        STRONG_SENTENCE_END_RE.search(current_segments[-1].text.strip())
    )
    hard_cap_exceeded = next_duration > (2 * TARGET_AUDIO_SECTION_DURATION_SEC)

    if gap_duration >= AUDIO_SECTION_BREAK_GAP_SEC:
        return True

    if (
        current_duration >= MIN_AUDIO_SECTION_DURATION_SEC
        and next_duration > TARGET_AUDIO_SECTION_DURATION_SEC
    ):
        return hard_cap_exceeded or ends_on_sentence_boundary

    return (
        current_duration >= MIN_AUDIO_SECTION_DURATION_SEC
        and current_chars >= MAX_AUDIO_SECTION_CHARS
        and (hard_cap_exceeded or ends_on_sentence_boundary)
    )


def _audio_section_from_segments(
    segments: list[TranscriptSegment], section_index: int
) -> ReportSection:
    title = _audio_title_from_segments(segments, fallback=f"Section {section_index}")
    return _section_from_segments(
        segments,
        section_index=section_index,
        title=title,
    )


def _interlude_section_from_segments(
    segments: list[TranscriptSegment],
    section_index: int,
    *,
    frame_id: str | None = None,
) -> ReportSection:
    title = f"Section {section_index}"
    return _section_from_segments(
        segments,
        section_index=section_index,
        title=title,
        frame_id=frame_id,
        is_interlude=True,
    )


def _section_from_segments(
    segments: list[TranscriptSegment],
    *,
    section_index: int,
    title: str,
    frame_id: str | None = None,
    is_interlude: bool = False,
) -> ReportSection:
    transcript_text = "\n\n".join(s for seg in segments if (s := seg.text.strip()))
    return ReportSection(
        id=f"section-{section_index}",
        title=title,
        start_sec=segments[0].start_sec,
        end_sec=segments[-1].end_sec,
        transcript_text=transcript_text,
        frame_id=frame_id,
        is_interlude=is_interlude,
    )


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


def _summary_score(segment: TranscriptSegment) -> float:
    text = segment.text.strip()
    words = _title_words(text)
    word_count = len(words)
    if word_count < 4:
        return -2.0

    informative_words = sum(1 for word in words[:14] if len(word) > 2)
    filler_words = sum(1 for word in words[:14] if word in TITLE_FILLER_WORDS)
    duration = max(0.0, segment.end_sec - segment.start_sec)
    score = informative_words + min(duration, 15.0) / 5.0
    score += _summary_noise_penalty(text)
    score += _summary_start_penalty(segment.start_sec)
    score += _summary_length_adjustment(word_count)
    score += _summary_punctuation_bonus(text)
    score += _summary_repetition_penalty(words)
    score += _summary_filler_penalty(filler_words, word_count)
    return score


def _action_item_score(segment: TranscriptSegment) -> float:
    text = segment.text.strip()
    words = _title_words(text)
    if len(words) < 2:
        return -1.0

    score = 2.0
    if SUMMARY_NOISE_PATTERN.search(text):
        score -= 3.0
    if segment.start_sec < 60.0:
        score -= 0.5
    if any(char in text for char in ".?!,:;"):
        score += 0.5
    return score


def _summary_noise_penalty(text: str) -> float:
    return -6.0 if SUMMARY_NOISE_PATTERN.search(text) else 0.0


def _summary_start_penalty(start_sec: float) -> float:
    if start_sec < 60.0:
        return -2.0
    if start_sec < 180.0:
        return -1.0
    return 0.0


def _summary_length_adjustment(word_count: int) -> float:
    return -1.5 if word_count > 28 else 0.0


def _summary_punctuation_bonus(text: str) -> float:
    return 1.0 if any(char in text for char in ".?!,:;") else 0.0


def _summary_repetition_penalty(words: list[str]) -> float:
    unique_ratio = len(set(words[:14])) / min(len(words), 14)
    if unique_ratio < 0.5:
        return -3.0
    if unique_ratio < 0.7:
        return -1.5
    return 0.0


def _summary_filler_penalty(filler_words: int, word_count: int) -> float:
    filler_ratio = filler_words / min(word_count, 14)
    if filler_ratio > 0.25:
        return -2.0
    if filler_ratio > 0.15:
        return -1.0
    return 0.0


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


def _sections_from_block(
    block: AlignmentBlock,
    *,
    block_segments: list[TranscriptSegment],
    interlude_ranges: list[tuple[float, float]],
    next_section_index: int,
) -> list[ReportSection]:
    if not block_segments:
        title = _title_from_text(block.transcript_text, fallback=f"Slide {next_section_index}")
        if block.title_hint:
            title = _title_from_text(block.title_hint, fallback=title)
        return [
            ReportSection(
                id=f"section-{next_section_index}",
                title=title,
                start_sec=block.start_sec,
                end_sec=block.end_sec,
                transcript_text=block.transcript_text,
                frame_id=block.frame_id,
            )
        ]

    runs: list[tuple[bool, list[TranscriptSegment]]] = []
    current_run: list[TranscriptSegment] = []
    current_is_interlude = _overlaps_interlude_ranges(block_segments[0], interlude_ranges)

    for segment in block_segments:
        segment_is_interlude = _overlaps_interlude_ranges(segment, interlude_ranges)
        if current_run and segment_is_interlude != current_is_interlude:
            runs.append((current_is_interlude, current_run))
            current_run = []
        current_run.append(segment)
        current_is_interlude = segment_is_interlude

    if current_run:
        runs.append((current_is_interlude, current_run))

    sections: list[ReportSection] = []
    for offset, (is_interlude, run_segments) in enumerate(runs):
        section_index = next_section_index + offset
        if is_interlude:
            sections.append(
                _interlude_section_from_segments(
                    run_segments,
                    section_index,
                    frame_id=block.frame_id,
                )
            )
            continue

        run_text = " ".join(segment.text for segment in run_segments)
        title = _title_from_text(
            run_text,
            fallback=f"Slide {section_index}",
        )
        if block.title_hint:
            title = _title_from_text(block.title_hint, fallback=title)
        sections.append(
            _section_from_segments(
                run_segments,
                section_index=section_index,
                title=title,
                frame_id=block.frame_id,
            )
        )

    return sections


def _render_interlude_sections(
    sections: list[ReportSection],
    *,
    detected_language: str | None,
) -> list[ReportSection]:
    rendered_sections: list[ReportSection] = []

    for section in sections:
        if not section.is_interlude:
            rendered_sections.append(section)
            continue

        rendered_sections.append(
            section.model_copy(
                update={
                    "title": _interlude_title(detected_language),
                    "transcript_text": _interlude_note(detected_language),
                    "is_interlude": True,
                }
            )
        )

    return rendered_sections


def _segments_excluding_interludes(
    segments: list[TranscriptSegment],
    sections: list[ReportSection],
) -> list[TranscriptSegment]:
    interlude_ranges = [
        (section.start_sec, section.end_sec) for section in sections if section.is_interlude
    ]
    if not interlude_ranges:
        return segments

    return [
        segment
        for segment in segments
        if not any(
            segment.start_sec < end_sec and segment.end_sec > start_sec
            for start_sec, end_sec in interlude_ranges
        )
    ]


def _detect_interlude_ranges(segments: list[TranscriptSegment]) -> list[tuple[float, float]]:
    meaningful_segments = [segment for segment in segments if segment.text.strip()]
    if not meaningful_segments:
        return []

    ranges: list[tuple[float, float]] = []
    current_run: list[TranscriptSegment] = []

    for segment in meaningful_segments:
        if not _is_likely_interlude_text(segment.text):
            _append_interlude_range(ranges, current_run)
            current_run = []
            continue

        if (
            current_run
            and (segment.start_sec - current_run[-1].end_sec) >= AUDIO_SECTION_BREAK_GAP_SEC
        ):
            _append_interlude_range(ranges, current_run)
            current_run = []
        current_run.append(segment)

    _append_interlude_range(ranges, current_run)
    return ranges


def _append_interlude_range(
    ranges: list[tuple[float, float]],
    segments: list[TranscriptSegment],
) -> None:
    if not segments:
        return
    if not any(_has_interlude_marker(segment.text) for segment in segments) and len(segments) < 2:
        return

    start_sec = segments[0].start_sec
    end_sec = segments[-1].end_sec
    ranges.append((start_sec, end_sec))


def _renderable_interlude_ranges(
    ranges: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    return [
        (start_sec, end_sec)
        for start_sec, end_sec in ranges
        if (end_sec - start_sec) >= MIN_INTERLUDE_DURATION_SEC
    ]


def _overlaps_interlude_ranges(
    segment: TranscriptSegment,
    interlude_ranges: list[tuple[float, float]],
) -> bool:
    return any(
        segment.start_sec < end_sec and segment.end_sec > start_sec
        for start_sec, end_sec in interlude_ranges
    )


def _is_likely_interlude_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _has_interlude_marker(stripped):
        return True

    words = INTERLUDE_WORD_RE.findall(stripped.casefold())
    original_words = INTERLUDE_WORD_RE.findall(stripped)
    if len(words) < INTERLUDE_MIN_WORDS:
        return False

    sample_size = min(len(words), 80)
    sampled_words = words[:sample_size]
    sampled_original_words = original_words[:sample_size]
    punctuation_density = sum(1 for char in stripped if char in ".?!:;") / sample_size
    unique_ratio = len(set(sampled_words)) / sample_size
    repeated_bigram_ratio = _repeated_bigram_ratio(sampled_words)
    noisy_word_ratio = _noisy_word_ratio(sampled_original_words)

    return bool(
        punctuation_density <= INTERLUDE_LOW_PUNCTUATION_DENSITY
        and (
            unique_ratio <= INTERLUDE_LOW_UNIQUE_RATIO
            or repeated_bigram_ratio >= INTERLUDE_REPEATED_BIGRAM_RATIO
            or noisy_word_ratio >= 0.35
        )
    )


def _has_interlude_marker(text: str) -> bool:
    return bool(INTERLUDE_MARKER_PATTERN.search(text))


def _repeated_bigram_ratio(words: list[str]) -> float:
    if len(words) < 4:
        return 0.0

    bigrams = list(pairwise(words))
    repeated_bigram_count = len(bigrams) - len(set(bigrams))
    return repeated_bigram_count / len(bigrams)


def _noisy_word_ratio(words: list[str]) -> float:
    noisy_words = sum(1 for word in words if _is_noisy_word(word))
    return noisy_words / len(words) if words else 0.0


def _is_noisy_word(word: str) -> bool:
    if len(word) < 5:
        return False
    if word.isupper():
        return False

    vowels = RUSSIAN_VOWELS | LATIN_VOWELS
    return not any(char in vowels for char in word.casefold())


def _interlude_title(detected_language: str | None) -> str:
    if detected_language == "ru":
        return "Музыкальная пауза"
    return "Music Interlude"


def _interlude_note(detected_language: str | None) -> str:
    if detected_language == "ru":
        return (
            "Музыкальная вставка или поэтический фрагмент. "
            "Исходная расшифровка сохранена в transcript.json."
        )
    return (
        "Music or spoken-performance interlude. The raw transcript is preserved in transcript.json."
    )
