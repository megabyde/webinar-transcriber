"""Utility helpers for optional cloud LLM integrations."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from webinar_transcriber.text_utils import split_paragraph_blocks

from .contracts import LLMConfigurationError, LLMProcessingError
from .prompts import REPORT_SECTION_EXCERPT_LIMIT

_SCHEMA_LABELS = {
    "SectionTextResponse": "Section polish",
    "ReportPolishResponse": "Report polish",
}
if TYPE_CHECKING:
    from webinar_transcriber.llm.schemas import ReportSectionUpdate
    from webinar_transcriber.models import ReportDocument


def required_provider_env(*, api_key_env: str, model_env: str) -> tuple[str, str]:
    """Return the required API key and model name for one provider.

    Returns:
        tuple[str, str]: The configured API key and model name.

    Raises:
        LLMConfigurationError: If either required environment variable is missing.
    """
    api_key = os.environ.get(api_key_env)
    model_name = os.environ.get(model_env)
    missing_vars = [
        env_name
        for env_name, value in ((api_key_env, api_key), (model_env, model_name))
        if not value
    ]
    if missing_vars:
        missing = ", ".join(missing_vars)
        raise LLMConfigurationError(f"Missing required LLM environment variables: {missing}.")
    return cast("tuple[str, str]", (api_key, model_name))


def build_report_polish_payload(
    report: ReportDocument,
    *,
    total_char_budget: int,
    section_transcripts: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the report-polish payload with a per-section excerpt budget.

    Returns:
        dict[str, object]: The structured payload sent to the metadata-polish stage.
    """
    section_count = max(len(report.sections), 1)
    per_section_budget = min(
        REPORT_SECTION_EXCERPT_LIMIT, max(200, total_char_budget // section_count)
    )

    return {
        "title": report.title,
        "source_file": report.source_file,
        "detected_language": report.detected_language,
        "current_summary": report.summary,
        "current_action_items": report.action_items,
        "sections": [
            {
                "id": section.id,
                "title": section.title,
                "start_sec": section.start_sec,
                "end_sec": section.end_sec,
                "transcript_excerpt": truncate_text(
                    (section_transcripts or {}).get(section.id, section.transcript_text),
                    per_section_budget,
                ),
            }
            for section in report.sections
        ],
    }


def normalize_report_lines(lines: Sequence[str], *, limit: int) -> list[str]:
    """Clean, dedupe, and cap a list of LLM-generated report lines.

    Returns:
        list[str]: The cleaned and deduplicated report lines.
    """
    normalized: list[str] = []
    seen: set[str] = set()

    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        dedupe_key = cleaned.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(cleaned)
        if len(normalized) == limit:
            break

    return normalized


def truncate_text(text: str, max_chars: int) -> str:
    """Trim text to one character budget while preserving whole-word readability.

    Returns:
        str: The truncated text.
    """
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 1].rstrip()}…"


def validated_section_titles(
    report: ReportDocument, section_titles: Sequence[ReportSectionUpdate]
) -> dict[str, str]:
    """Validate returned section-title updates against the existing report.

    Returns:
        dict[str, str]: Cleaned replacement titles keyed by section ID.

    Raises:
        LLMProcessingError: If the response references invalid, duplicate, or empty titles.
    """
    valid_ids = {section.id for section in report.sections}
    polished_titles: dict[str, str] = {}

    for item in section_titles:
        if item.id not in valid_ids:
            raise LLMProcessingError("Report polish response returned an unknown section ID.")
        if item.id in polished_titles:
            raise LLMProcessingError("Report polish response returned duplicate section IDs.")
        cleaned_title = item.title.strip()
        if not cleaned_title:
            raise LLMProcessingError("Report polish response returned an empty section title.")
        polished_titles[item.id] = cleaned_title

    return polished_titles


def normalize_polished_text(*, polished_text: str) -> str:
    """Normalize multi-paragraph LLM text.

    Returns:
        str: The normalized polished text.
    """
    cleaned = polished_text.strip()
    if not cleaned:
        return ""

    cleaned = _normalize_llm_paragraphs(cleaned)
    return re.sub(r"[ \t]+\n", "\n", cleaned)


def normalize_polished_section_text(*, original_text: str, polished_text: str) -> str:
    """Normalize polished section text.

    Returns:
        str: The accepted polished section text, or the original text when empty.
    """
    cleaned = normalize_polished_text(polished_text=polished_text)
    if not cleaned:
        return original_text
    return cleaned


def normalize_polished_section_tldr(tldr: str) -> str:
    """Normalize a polished TL;DR into stable paragraph spacing.

    Returns:
        str: The normalized TL;DR text.
    """
    cleaned = tldr.strip()
    if not cleaned:
        return ""

    return _normalize_tldr_blocks(cleaned)


def _normalize_tldr_blocks(text: str) -> str:
    normalized = _normalize_llm_paragraphs(text)
    return re.sub(r"(?<=\S)[ \t]+(?=(?:[-*]|\d+[.)])\s+)", "\n", normalized)


def _normalize_llm_paragraphs(text: str) -> str:
    """Normalize LLM paragraph blocks into stable spacing.

    Returns:
        str: The normalized paragraph text.
    """
    return "\n\n".join(
        split_paragraph_blocks(text, flexible_blank_lines=True, normalize_inline_whitespace=True)
    )


def extract_response_metadata(response: object) -> dict[str, object]:
    """Extract non-sensitive provider response metadata for diagnostics."""
    metadata: dict[str, object] = {}
    choice = _first_choice(response)
    message = getattr(choice, "message", None) if choice is not None else None

    if finish_reason := _text_attr(choice, "finish_reason"):
        metadata["finish_reason"] = finish_reason
    if stop_reason := _text_attr(response, "stop_reason"):
        metadata["stop_reason"] = stop_reason
    refusal = getattr(message, "refusal", None)
    if refusal:
        metadata["refusal"] = True

    safety = _json_safe(getattr(choice, "content_filter_results", None))
    if safety is not None:
        metadata["safety"] = safety
    prompt_filter_results = _json_safe(getattr(response, "prompt_filter_results", None))
    if prompt_filter_results is not None:
        metadata["prompt_filter_results"] = prompt_filter_results

    return metadata


def _first_choice(response: object) -> object | None:
    choices = getattr(response, "choices", None)
    if isinstance(choices, Sequence) and choices:
        return choices[0]
    return None


def _text_attr(value: object | None, name: str) -> str | None:
    text = getattr(value, name, None)
    return text if isinstance(text, str) and text else None


def _json_safe(value: object) -> object | None:
    try:
        return json.loads(json.dumps(value, default=_json_default, ensure_ascii=False))
    except (TypeError, ValueError):
        return None


def _json_default(value: object) -> object:
    if callable(model_dump := getattr(value, "model_dump", None)):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    if callable(to_dict := getattr(value, "to_dict", None)):
        return to_dict()
    return None


def schema_label(response_model: type[object]) -> str:
    """Return the human-facing label for one structured response schema."""
    return _SCHEMA_LABELS.get(response_model.__name__, "Structured LLM")
