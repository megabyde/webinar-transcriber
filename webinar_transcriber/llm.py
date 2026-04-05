"""Optional cloud LLM helpers for transcript and report enhancement."""

from __future__ import annotations

import importlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import openai
from pydantic import BaseModel, Field

from webinar_transcriber.models import ReportDocument, ReportSection
from webinar_transcriber.usage import merge_usage, merge_usage_into

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

SchemaModelT = TypeVar("SchemaModelT", bound=BaseModel)


REPORT_POLISH_TOTAL_CHAR_BUDGET = 16_000
REPORT_SECTION_EXCERPT_LIMIT = 1_200
SUMMARY_ITEM_LIMIT = 5
ACTION_ITEM_LIMIT = 7
SECTION_POLISH_MAX_WORKERS = 6
REPORT_POLISH_SYSTEM_PROMPT = """
You are improving a structured report built from an automatic speech transcript.

Keep the original language. Do not translate. Preserve meaning, names, and terminology.
Improve clarity without adding facts or interpretation.
Return factual summary bullets, concrete action items when they were directly assigned,
strongly implied, or presented as practical recommended next steps by the speakers,
and better section titles.
Do not turn general themes, broad observations, or abstract best practices into action items.
If there are no clear action items, return an empty list.
Do not change section IDs.
""".strip()
SECTION_POLISH_SYSTEM_PROMPT = """
You are cleaning one section of an automatic speech transcript.

Keep the original language. Do not translate. Preserve names, terminology, and meaning.
Fix punctuation, capitalization, and obvious ASR mistakes.
Preserve meaning, order, and level of detail.
Apply only light rephrasing for readability.
Do not add new facts, interpretations, advice, or commentary.
Prefer normal sentence punctuation. Do not add stylistic ellipses unless the source
clearly trails off.
Split the text into natural paragraphs separated by blank lines, usually 3-6 sentences
per paragraph. Insert a paragraph break when the speaker shifts to a new subpoint or
topic. Avoid returning one giant paragraph unless the input is extremely short.
Also return a factual section cheat sheet / TL;DR that is longer and more informative than a
one-line summary. Usually write 3-6 short paragraphs, and you may use bullets or numbered items
when that is clearer and more compact. Capture the main claims, important examples, caveats,
concrete mechanisms, and practical takeaways when they are present in the source. Prefer a format
that is easy to scan quickly without turning into a wall of text. The cheat sheet should be easier
to read than the transcript, but it must not add new facts.
""".strip()


class LLMConfigurationError(RuntimeError):
    """Raised when required LLM configuration is missing."""


class LLMProcessingError(RuntimeError):
    """Raised when the LLM response cannot be validated or applied."""


@dataclass(frozen=True)
class LLMReportPolishResult:
    """Validated result from the report-polishing LLM stage."""

    summary: list[str]
    action_items: list[str]
    section_titles: dict[str, str]
    section_tldrs: dict[str, str]
    section_transcripts: dict[str, str]
    usage: dict[str, int]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LLMSectionPolishResult:
    """Validated result from the section-text polishing phase."""

    section_tldrs: dict[str, str]
    section_transcripts: dict[str, str]
    usage: dict[str, int]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LLMReportMetadataResult:
    """Validated result from the final report metadata polishing phase."""

    summary: list[str]
    action_items: list[str]
    section_titles: dict[str, str]
    usage: dict[str, int]


@dataclass(frozen=True)
class _SectionPolishOutputs:
    transcripts: dict[str, str]
    tldrs: dict[str, str]


@dataclass(frozen=True)
class _ProviderEnvConfig:
    provider_name: str
    api_key_env: str
    model_env: str


class SectionTextResponse(BaseModel):
    """Structured LLM response for one polished section body."""

    tldr: str = ""
    transcript_text: str = ""


class ReportSectionUpdate(BaseModel):
    """Replacement content for one report section."""

    id: str
    title: str


class ReportPolishResponse(BaseModel):
    """Structured LLM response for report polishing."""

    summary: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    section_updates: list[ReportSectionUpdate] = Field(default_factory=list)


@dataclass(frozen=True)
class LLMReportPolishPlan:
    """Execution plan for report polishing progress/reporting."""

    section_count: int
    worker_count: int


class LLMProcessor(Protocol):
    """Protocol for optional transcript/report enhancement backends."""

    @property
    def provider_name(self) -> str:
        """Return the configured provider identifier."""

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""

    def polish_report(self, report: ReportDocument) -> LLMReportPolishResult:
        """Return polished summary, action items, and section titles."""

    def polish_report_sections_with_progress(
        self,
        report: ReportDocument,
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> LLMSectionPolishResult:
        """Return polished section text with per-section progress updates."""

    def polish_report_metadata(
        self,
        report: ReportDocument,
        *,
        section_transcripts: dict[str, str],
    ) -> LLMReportMetadataResult:
        """Return polished summary, action items, and section titles."""

    def polish_report_with_progress(
        self,
        report: ReportDocument,
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> LLMReportPolishResult:
        """Return polished report fields with per-section progress updates."""

    def report_polish_plan(self, report: ReportDocument) -> LLMReportPolishPlan:
        """Return concurrency details for report polishing."""


class _BaseLLMProcessor:
    """Shared report-polishing flow for structured LLM backends."""

    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str,
        section_max_workers: int = SECTION_POLISH_MAX_WORKERS,
        report_char_budget: int = REPORT_POLISH_TOTAL_CHAR_BUDGET,
    ) -> None:
        self._provider_name = provider_name
        self._model_name = model_name
        self._report_char_budget = report_char_budget
        self._section_max_workers = section_max_workers

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Return the configured LLM provider identifier."""
        return self._provider_name

    def polish_report(self, report: ReportDocument) -> LLMReportPolishResult:
        """Polish section text, summary, action items, and section titles."""
        return self.polish_report_with_progress(report)

    def polish_report_sections_with_progress(
        self,
        report: ReportDocument,
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> LLMSectionPolishResult:
        """Polish section transcript text with per-section progress updates."""
        usage_totals: dict[str, int] = {}
        warnings: list[str] = []
        polished_section_texts = self._polish_section_texts(
            report,
            progress_callback=progress_callback,
            usage_totals=usage_totals,
            warnings=warnings,
        )
        return LLMSectionPolishResult(
            section_tldrs=polished_section_texts.tldrs,
            section_transcripts=polished_section_texts.transcripts,
            usage=usage_totals,
            warnings=warnings,
        )

    def polish_report_metadata(
        self,
        report: ReportDocument,
        *,
        section_transcripts: dict[str, str],
    ) -> LLMReportMetadataResult:
        """Polish report summary, action items, and section titles."""
        polished_report = ReportDocument(
            title=report.title,
            source_file=report.source_file,
            media_type=report.media_type,
            detected_language=report.detected_language,
            summary=list(report.summary),
            action_items=list(report.action_items),
            sections=[
                ReportSection(
                    id=section.id,
                    title=section.title,
                    start_sec=section.start_sec,
                    end_sec=section.end_sec,
                    tldr=section.tldr,
                    transcript_text=section_transcripts.get(section.id, section.transcript_text),
                    bullet_points=list(section.bullet_points),
                    frame_id=section.frame_id,
                    image_path=section.image_path,
                    is_interlude=section.is_interlude,
                )
                for section in report.sections
            ],
            warnings=list(report.warnings),
        )
        payload = _build_report_polish_payload(
            polished_report,
            total_char_budget=self._report_char_budget,
        )
        parsed, usage = self._parse_structured_response(
            system_prompt=REPORT_POLISH_SYSTEM_PROMPT,
            user_payload=payload,
            response_model=ReportPolishResponse,
            error_prefix="Report polishing failed",
        )

        return LLMReportMetadataResult(
            summary=_normalize_report_lines(parsed.summary, limit=SUMMARY_ITEM_LIMIT),
            action_items=_normalize_report_lines(
                parsed.action_items,
                limit=ACTION_ITEM_LIMIT,
            ),
            section_titles=_validated_section_titles(report, parsed.section_updates),
            usage=usage,
        )

    def polish_report_with_progress(
        self,
        report: ReportDocument,
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> LLMReportPolishResult:
        """Polish section text first, then polish report summary and section titles."""
        section_result = self.polish_report_sections_with_progress(
            report,
            progress_callback=progress_callback,
        )
        metadata_result = self.polish_report_metadata(
            report,
            section_transcripts=section_result.section_transcripts,
        )
        usage_totals = merge_usage(section_result.usage, metadata_result.usage)

        return LLMReportPolishResult(
            summary=metadata_result.summary,
            action_items=metadata_result.action_items,
            section_titles=metadata_result.section_titles,
            section_tldrs=section_result.section_tldrs,
            section_transcripts=section_result.section_transcripts,
            usage=usage_totals,
            warnings=section_result.warnings,
        )

    def report_polish_plan(self, report: ReportDocument) -> LLMReportPolishPlan:
        """Return concurrency details for report polishing."""
        return LLMReportPolishPlan(
            section_count=len(report.sections),
            worker_count=min(self._section_max_workers, max(len(report.sections), 1)),
        )

    def _polish_section_texts(
        self,
        report: ReportDocument,
        *,
        progress_callback: Callable[[int], None] | None,
        usage_totals: dict[str, int],
        warnings: list[str],
    ) -> _SectionPolishOutputs:
        plan = self.report_polish_plan(report)
        if not report.sections:
            return _SectionPolishOutputs(transcripts={}, tldrs={})

        polished_transcripts: dict[str, str] = {}
        polished_tldrs: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=plan.worker_count) as executor:
            future_to_section = {
                executor.submit(self._polish_section_text, section): section
                for section in report.sections
                if not section.is_interlude
            }
            skipped_sections = [section for section in report.sections if section.is_interlude]
            for section in skipped_sections:
                polished_transcripts[section.id] = section.transcript_text
                if section.tldr:
                    polished_tldrs[section.id] = section.tldr
                warnings.append(
                    f"Skipped LLM section polish for likely music/interlude section {section.id}."
                )
                if progress_callback is not None:
                    progress_callback(1)
            for future in as_completed(future_to_section):
                section = future_to_section[future]
                try:
                    transcript_text, tldr, usage, section_warnings = future.result()
                except LLMProcessingError as error:
                    transcript_text = section.transcript_text
                    tldr = section.tldr or ""
                    usage = {}
                    section_warnings = [str(error)]
                polished_transcripts[section.id] = transcript_text
                if tldr:
                    polished_tldrs[section.id] = tldr
                merge_usage_into(usage_totals, usage)
                warnings.extend(section_warnings)
                if progress_callback is not None:
                    progress_callback(1)

        return _SectionPolishOutputs(
            transcripts=polished_transcripts,
            tldrs=polished_tldrs,
        )

    def _polish_section_text(
        self,
        section: ReportSection,
    ) -> tuple[str, str, dict[str, int], list[str]]:
        payload = {
            "id": section.id,
            "title": section.title,
            "start_sec": section.start_sec,
            "end_sec": section.end_sec,
            "transcript_text": section.transcript_text,
        }
        parsed, usage = self._parse_structured_response(
            system_prompt=SECTION_POLISH_SYSTEM_PROMPT,
            user_payload=payload,
            response_model=SectionTextResponse,
            error_prefix=f"Section polishing failed for {section.id}",
        )

        transcript_text = _normalize_polished_section_text(
            original_text=section.transcript_text,
            polished_text=parsed.transcript_text,
            section_id=section.id,
        )
        tldr = _normalize_polished_section_tldr(parsed.tldr)
        warnings: list[str] = []
        if transcript_text == section.transcript_text and not parsed.transcript_text.strip():
            warnings.append(
                f"Section polish response returned an empty transcript text for {section.id}; "
                "kept original text."
            )

        return transcript_text, tldr, usage, warnings

    def _parse_structured_response(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, object],
        response_model: type[SchemaModelT],
        error_prefix: str,
    ) -> tuple[SchemaModelT, dict[str, int]]:
        raise NotImplementedError


class OpenAILLMProcessor(_BaseLLMProcessor):
    """OpenAI-backed transcript polishing integration."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        section_max_workers: int = SECTION_POLISH_MAX_WORKERS,
        report_char_budget: int = REPORT_POLISH_TOTAL_CHAR_BUDGET,
    ) -> None:
        super().__init__(
            provider_name="openai",
            model_name=model_name,
            section_max_workers=section_max_workers,
            report_char_budget=report_char_budget,
        )
        self._client = openai.OpenAI(api_key=api_key)

    def _parse_structured_response(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, object],
        response_model: type[SchemaModelT],
        error_prefix: str,
    ) -> tuple[SchemaModelT, dict[str, int]]:
        try:
            response = self._client.responses.parse(
                model=self.model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                text_format=response_model,
            )
        except Exception as error:  # pragma: no cover - backend-specific SDK errors
            raise LLMProcessingError(f"{error_prefix}: {error}") from error

        parsed = getattr(response, "output_parsed", None)
        if not isinstance(parsed, response_model):
            raise LLMProcessingError(
                f"{_schema_label(response_model)} response did not match the schema."
            )
        return parsed, _extract_usage(response)


class AnthropicLLMProcessor(_BaseLLMProcessor):
    """Anthropic-backed transcript polishing integration."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        section_max_workers: int = SECTION_POLISH_MAX_WORKERS,
        report_char_budget: int = REPORT_POLISH_TOTAL_CHAR_BUDGET,
    ) -> None:
        super().__init__(
            provider_name="anthropic",
            model_name=model_name,
            section_max_workers=section_max_workers,
            report_char_budget=report_char_budget,
        )
        self._client = _build_anthropic_client(api_key)

    def _parse_structured_response(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, object],
        response_model: type[SchemaModelT],
        error_prefix: str,
    ) -> tuple[SchemaModelT, dict[str, int]]:
        prompt = _anthropic_structured_prompt(
            system_prompt=system_prompt,
            user_payload=user_payload,
            response_model=response_model,
        )
        try:
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as error:  # pragma: no cover - backend-specific SDK errors
            raise LLMProcessingError(f"{error_prefix}: {error}") from error

        text = _anthropic_response_text(response)
        try:
            parsed = response_model.model_validate_json(_extract_json_text(text))
        except Exception as error:
            raise LLMProcessingError(
                f"{_schema_label(response_model)} response did not match the schema."
            ) from error
        return parsed, _extract_usage(response)


def build_llm_processor_from_env() -> LLMProcessor:
    """Build a configured LLM processor from environment variables."""
    provider = os.environ.get("LLM_PROVIDER", "openai").strip().casefold()
    match provider:
        case "openai":
            api_key, model_name = _required_provider_env(
                _ProviderEnvConfig(
                    provider_name="openai",
                    api_key_env="OPENAI_API_KEY",
                    model_env="OPENAI_MODEL",
                )
            )
            return OpenAILLMProcessor(api_key=api_key, model_name=model_name)
        case "anthropic":
            api_key, model_name = _required_provider_env(
                _ProviderEnvConfig(
                    provider_name="anthropic",
                    api_key_env="ANTHROPIC_API_KEY",
                    model_env="ANTHROPIC_MODEL",
                )
            )
            return AnthropicLLMProcessor(api_key=api_key, model_name=model_name)
        case _:
            raise LLMConfigurationError(
                "Unsupported LLM provider. Set LLM_PROVIDER to 'openai' or 'anthropic'."
            )


def _build_anthropic_client(api_key: str) -> Any:
    return _build_sdk_client(
        module_name="anthropic",
        client_class_name="Anthropic",
        api_key=api_key,
        missing_sdk_message=(
            "LLM requested but the Anthropic SDK is not installed in this environment."
        ),
    )


def _build_sdk_client(
    *,
    module_name: str,
    client_class_name: str,
    api_key: str,
    missing_sdk_message: str,
) -> Any:
    try:
        sdk_module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:  # pragma: no cover - dependency wiring only
        raise LLMConfigurationError(missing_sdk_message) from error
    client_class = getattr(sdk_module, client_class_name)
    return client_class(api_key=api_key)


def _required_provider_env(config: _ProviderEnvConfig) -> tuple[str, str]:
    api_key = os.environ.get(config.api_key_env)
    model_name = os.environ.get(config.model_env)
    missing_vars = [
        env_name
        for env_name, value in (
            (config.api_key_env, api_key),
            (config.model_env, model_name),
        )
        if not value
    ]
    if missing_vars:
        missing = ", ".join(missing_vars)
        raise LLMConfigurationError(f"Missing required LLM environment variables: {missing}.")
    assert api_key is not None
    assert model_name is not None
    return api_key, model_name


def _build_report_polish_payload(
    report: ReportDocument,
    *,
    total_char_budget: int,
) -> dict[str, object]:
    section_count = max(len(report.sections), 1)
    per_section_budget = min(
        REPORT_SECTION_EXCERPT_LIMIT,
        max(200, total_char_budget // section_count),
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
                "transcript_excerpt": _truncate_text(section.transcript_text, per_section_budget),
            }
            for section in report.sections
        ],
    }


def _normalize_report_lines(lines: Sequence[str], *, limit: int) -> list[str]:
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


def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 1].rstrip()}…"


def _validated_section_titles(
    report: ReportDocument,
    section_titles: Sequence[ReportSectionUpdate],
) -> dict[str, str]:
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


def _normalize_polished_text(*, original_text: str, polished_text: str) -> str:
    cleaned = polished_text.strip()
    if not cleaned:
        return ""

    paragraphs = [
        re.sub(r"\s+", " ", p) for block in re.split(r"\n\s*\n+", cleaned) if (p := block.strip())
    ]
    cleaned = "\n\n".join(paragraphs)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)

    if not re.search(r"(?:\.{3}|…)\s*$", original_text.strip()):
        cleaned = re.sub(r"(?:\s*)(?:\.{3}|…)+\s*$", ".", cleaned)

    return cleaned


def _normalize_polished_section_text(
    *,
    original_text: str,
    polished_text: str,
    section_id: str,
) -> str:
    cleaned = _normalize_polished_text(
        original_text=original_text,
        polished_text=polished_text,
    )
    if not cleaned:
        return original_text
    if len(cleaned) < 20 and len(original_text.strip()) > 100:
        raise LLMProcessingError(
            f"Section polish response looked truncated for {section_id}; kept original text."
        )
    return cleaned


def _normalize_polished_section_tldr(tldr: str) -> str:
    cleaned = tldr.strip()
    if not cleaned:
        return ""

    paragraphs = [
        re.sub(r"\s+", " ", p) for block in re.split(r"\n\s*\n+", cleaned) if (p := block.strip())
    ]
    return "\n\n".join(paragraphs)


def _extract_usage(response: object) -> dict[str, int]:
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


def _anthropic_structured_prompt(
    *,
    system_prompt: str,
    user_payload: Mapping[str, object],
    response_model: type[BaseModel],
) -> str:
    schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
    payload = json.dumps(user_payload, ensure_ascii=False)
    return (
        f"{system_prompt}\n\n"
        "Return only a JSON object that matches this JSON Schema exactly.\n"
        f"{schema}\n\n"
        "User payload:\n"
        f"{payload}"
    )


def _anthropic_response_text(response: object) -> str:
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


def _extract_json_text(text: str) -> str:
    cleaned = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fenced is not None:
        return fenced.group(1)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _schema_label(response_model: type[BaseModel]) -> str:
    if response_model is SectionTextResponse:
        return "Section polish"
    if response_model is ReportPolishResponse:
        return "Report polish"
    return "Structured LLM"
