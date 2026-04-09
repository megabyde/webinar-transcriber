"""Shared models and helpers for optional cloud LLM integrations."""

from .contracts import (
    LLMConfigurationError,
    LLMProcessingError,
    LLMProcessor,
    LLMReportMetadataResult,
    LLMReportPolishPlan,
    LLMReportPolishResult,
    LLMSectionPolishResult,
    ReportPolishResponse,
    ReportSectionUpdate,
    SchemaModelT,
    SectionTextResponse,
)
from .flow import _BaseLLMProcessor
from .prompts import (
    ACTION_ITEM_LIMIT,
    REPORT_POLISH_SYSTEM_PROMPT,
    REPORT_POLISH_TOTAL_CHAR_BUDGET,
    SECTION_POLISH_MAX_WORKERS,
    SECTION_POLISH_SYSTEM_PROMPT,
    SUMMARY_ITEM_LIMIT,
)
from .utils import (
    anthropic_response_text,
    anthropic_structured_prompt,
    build_report_polish_payload,
    extract_json_text,
    extract_usage,
    normalize_polished_section_text,
    normalize_polished_section_tldr,
    normalize_polished_text,
    normalize_report_lines,
    required_provider_env,
    schema_label,
    truncate_text,
    validated_section_titles,
)

_build_report_polish_payload = build_report_polish_payload
_normalize_report_lines = normalize_report_lines
_truncate_text = truncate_text
_validated_section_titles = validated_section_titles
_normalize_polished_text = normalize_polished_text
_normalize_polished_section_text = normalize_polished_section_text
_normalize_polished_section_tldr = normalize_polished_section_tldr
_extract_usage = extract_usage
_anthropic_structured_prompt = anthropic_structured_prompt
_anthropic_response_text = anthropic_response_text
_extract_json_text = extract_json_text
_schema_label = schema_label
_required_provider_env = required_provider_env

__all__ = [
    "ACTION_ITEM_LIMIT",
    "REPORT_POLISH_SYSTEM_PROMPT",
    "REPORT_POLISH_TOTAL_CHAR_BUDGET",
    "SECTION_POLISH_MAX_WORKERS",
    "SECTION_POLISH_SYSTEM_PROMPT",
    "SUMMARY_ITEM_LIMIT",
    "LLMConfigurationError",
    "LLMProcessingError",
    "LLMProcessor",
    "LLMReportMetadataResult",
    "LLMReportPolishPlan",
    "LLMReportPolishResult",
    "LLMSectionPolishResult",
    "ReportPolishResponse",
    "ReportSectionUpdate",
    "SchemaModelT",
    "SectionTextResponse",
    "_BaseLLMProcessor",
    "_anthropic_response_text",
    "_anthropic_structured_prompt",
    "_build_report_polish_payload",
    "_extract_json_text",
    "_extract_usage",
    "_normalize_polished_section_text",
    "_normalize_polished_section_tldr",
    "_normalize_polished_text",
    "_normalize_report_lines",
    "_required_provider_env",
    "_schema_label",
    "_truncate_text",
    "_validated_section_titles",
]
