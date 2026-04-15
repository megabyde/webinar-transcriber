"""Utility helpers for optional cloud LLM integrations."""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

from .contracts import (
    LLMConfigurationError,
    LLMProcessingError,
    ReportPolishResponse,
    ReportSectionUpdate,
    SectionTextResponse,
)
from .prompts import REPORT_SECTION_EXCERPT_LIMIT

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pydantic import BaseModel

    from webinar_transcriber.models import ReportDocument


def required_provider_env(*, api_key_env: str, model_env: str) -> tuple[str, str]:
    """Return the required API key and model name for one provider."""
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
    assert api_key is not None
    assert model_name is not None
    return api_key, model_name


def build_report_polish_payload(
    report: ReportDocument,
    *,
    total_char_budget: int,
    section_transcripts: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the report-polish payload with a per-section excerpt budget."""
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
    """Clean, dedupe, and cap a list of LLM-generated report lines."""
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
    """Trim text to one character budget while preserving whole-word readability."""
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 1].rstrip()}…"


def validated_section_titles(
    report: ReportDocument, section_titles: Sequence[ReportSectionUpdate]
) -> dict[str, str]:
    """Validate returned section-title updates against the existing report."""
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


def normalize_polished_text(*, original_text: str, polished_text: str) -> str:
    """Normalize multi-paragraph LLM text while preserving trailing-finality semantics."""
    cleaned = polished_text.strip()
    if not cleaned:
        return ""

    cleaned = _normalize_paragraph_blocks(cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)

    if not re.search(r"(?:\.{3}|…)\s*$", original_text.strip()):
        cleaned = re.sub(r"(?:\s*)(?:\.{3}|…)+\s*$", ".", cleaned)

    return cleaned


def normalize_polished_section_text(
    *, original_text: str, polished_text: str, section_id: str
) -> str:
    """Normalize polished section text and reject suspiciously truncated output."""
    cleaned = normalize_polished_text(original_text=original_text, polished_text=polished_text)
    if not cleaned:
        return original_text
    if len(cleaned) < 20 and len(original_text.strip()) > 100:
        raise LLMProcessingError(
            f"Section polish response looked truncated for {section_id}; kept original text."
        )
    return cleaned


def normalize_polished_section_tldr(tldr: str) -> str:
    """Normalize a polished TL;DR into stable paragraph spacing."""
    cleaned = tldr.strip()
    if not cleaned:
        return ""

    return _normalize_paragraph_blocks(cleaned)


def _normalize_paragraph_blocks(text: str) -> str:
    """Normalize blank-line paragraph blocks into stable spacing."""
    paragraphs = [
        re.sub(r"\s+", " ", p) for block in re.split(r"\n\s*\n+", text) if (p := block.strip())
    ]
    return "\n\n".join(paragraphs)


def extract_usage(response: object) -> dict[str, int]:
    """Extract token usage from provider responses with a stable key subset."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {
            key: value
            for key, value in usage.items()
            if isinstance(value, int) and key in {"input_tokens", "output_tokens", "total_tokens"}
        }

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


def anthropic_structured_prompt(
    *, system_prompt: str, user_payload: Mapping[str, object], response_model: type[BaseModel]
) -> str:
    """Build the Anthropic prompt that embeds the target JSON schema."""
    schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
    payload = json.dumps(user_payload, ensure_ascii=False)
    return (
        f"{system_prompt}\n\n"
        "Return only a JSON object that matches this JSON Schema exactly.\n"
        f"{schema}\n\n"
        "User payload:\n"
        f"{payload}"
    )


def anthropic_response_text(response: object) -> str:
    """Extract concatenated text content from an Anthropic response object."""
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        raise LLMProcessingError("Anthropic response did not contain text content.")

    text_chunks = [
        block.text
        for block in content
        if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)
    ]
    if not text_chunks:
        raise LLMProcessingError("Anthropic response did not contain text content.")
    return "".join(text_chunks).strip()


def extract_json_text(text: str) -> str:
    """Extract the first JSON object from plain or fenced text."""
    cleaned = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fenced is not None:
        return fenced.group(1)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def schema_label(response_model: type[BaseModel]) -> str:
    """Return the human-facing label for one structured response schema."""
    if response_model is SectionTextResponse:
        return "Section polish"
    if response_model is ReportPolishResponse:
        return "Report polish"
    return "Structured LLM"
