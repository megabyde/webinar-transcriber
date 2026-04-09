"""Optional cloud LLM helpers for transcript and report enhancement."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import anthropic
import openai

if TYPE_CHECKING:
    from collections.abc import Mapping

from webinar_transcriber.llm_shared import (
    ACTION_ITEM_LIMIT,
    REPORT_POLISH_TOTAL_CHAR_BUDGET,
    SECTION_POLISH_MAX_WORKERS,
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
    _anthropic_response_text,
    _anthropic_structured_prompt,
    _BaseLLMProcessor,
    _extract_json_text,
    _extract_usage,
    _ProviderEnvConfig,
    _required_provider_env,
    _schema_label,
)


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
    """Construct the Anthropic client for the configured API key."""
    return anthropic.Anthropic(api_key=api_key)


__all__ = [
    "ACTION_ITEM_LIMIT",
    "REPORT_POLISH_TOTAL_CHAR_BUDGET",
    "SECTION_POLISH_MAX_WORKERS",
    "AnthropicLLMProcessor",
    "LLMConfigurationError",
    "LLMProcessingError",
    "LLMProcessor",
    "LLMReportMetadataResult",
    "LLMReportPolishPlan",
    "LLMReportPolishResult",
    "LLMSectionPolishResult",
    "OpenAILLMProcessor",
    "ReportPolishResponse",
    "ReportSectionUpdate",
    "SectionTextResponse",
    "build_llm_processor_from_env",
]
