"""Anthropic-backed optional cloud LLM integration."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .contracts import LLMProcessingError, SchemaModelT
from .flow import _BaseLLMProcessor
from .prompts import REPORT_POLISH_TOTAL_CHAR_BUDGET, SECTION_POLISH_MAX_WORKERS
from .utils import (
    anthropic_response_text,
    anthropic_structured_prompt,
    extract_json_text,
    extract_usage,
    schema_label,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


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
        """Initialize the Anthropic-backed report polisher."""
        super().__init__(
            provider_name="anthropic",
            model_name=model_name,
            section_max_workers=section_max_workers,
            report_char_budget=report_char_budget,
        )
        self._client = importlib.import_module("anthropic").Anthropic(api_key=api_key)

    def _parse_structured_response(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, object],
        response_model: type[SchemaModelT],
        error_prefix: str,
    ) -> tuple[SchemaModelT, dict[str, int]]:
        prompt = anthropic_structured_prompt(
            system_prompt=system_prompt, user_payload=user_payload, response_model=response_model
        )
        try:
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as error:  # pragma: no cover - backend-specific SDK errors
            raise LLMProcessingError(f"{error_prefix}: {error}") from error

        text = anthropic_response_text(response)
        try:
            parsed = response_model.model_validate_json(extract_json_text(text))
        except Exception as error:
            raise LLMProcessingError(
                f"{schema_label(response_model)} response did not match the schema."
            ) from error
        return parsed, extract_usage(response)
