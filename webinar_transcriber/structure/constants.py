"""Shared constants and regex helpers for report structuring."""

import re


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
INTERLUDE_LOW_UNIQUE_RATIO = 0.5
TITLE_FILLER_WORDS = {
    "first",
    "just",
    "like",
    "okay",
    "right",
    "so",
    "then",
    "well",
    "вот",
    "ну",
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
