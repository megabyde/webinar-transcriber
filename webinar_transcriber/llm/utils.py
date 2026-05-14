"""Utility helpers for optional cloud LLM integrations."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from .contracts import (
    LLMConfigurationError,
    LLMProcessingError,
)
from .prompts import REPORT_SECTION_EXCERPT_LIMIT

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

    cleaned = _normalize_paragraph_blocks(cleaned)
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
    normalized = _normalize_paragraph_blocks(text)
    return re.sub(r"(?<=\S)[ \t]+(?=(?:[-*]|\d+[.)])\s+)", "\n", normalized)


def _normalize_paragraph_blocks(text: str) -> str:
    """Normalize blank-line paragraph blocks into stable spacing.

    Returns:
        str: The normalized paragraph text.
    """
    paragraphs = []
    for block in re.split(r"\n\s*\n+", text):
        lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in block.strip().splitlines()]
        paragraph = "\n".join(line for line in lines if line)
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def extract_usage(response: object) -> dict[str, int]:
    """Extract token usage from provider responses with a stable key subset.

    Returns:
        dict[str, int]: The extracted token counts.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    extracted: dict[str, int] = {}
    for field_name in ("input_tokens", "output_tokens", "total_tokens"):
        field_value = getattr(usage, field_name, None)
        if isinstance(field_value, int):
            extracted[field_name] = field_value
    if (
        "total_tokens" not in extracted
        and "input_tokens" in extracted
        and "output_tokens" in extracted
    ):
        extracted["total_tokens"] = extracted["input_tokens"] + extracted["output_tokens"]
    return extracted


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
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): safe_value
            for key, item in value.items()
            if (safe_value := _json_safe(item)) is not None
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [item for item in (_json_safe(item) for item in value) if item is not None]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    return None


def schema_label(response_model: type[object]) -> str:
    """Return the human-facing label for one structured response schema."""
    if response_model.__name__ == "SectionTextResponse":
        return "Section polish"
    if response_model.__name__ == "ReportPolishResponse":
        return "Report polish"
    return "Structured LLM"
