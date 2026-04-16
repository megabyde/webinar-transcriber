"""OpenAI-backed optional cloud LLM integration."""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

from .contracts import LLMProcessingError, SchemaModelT
from .flow import _BaseLLMProcessor
from .prompts import REPORT_POLISH_TOTAL_CHAR_BUDGET, SECTION_POLISH_MAX_WORKERS
from .utils import extract_usage, schema_label

if TYPE_CHECKING:
    from collections.abc import Mapping


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
        """Initialize the OpenAI-backed report polisher."""
        super().__init__(
            provider_name="openai",
            model_name=model_name,
            section_max_workers=section_max_workers,
            report_char_budget=report_char_budget,
        )
        self._client = importlib.import_module("openai").OpenAI(api_key=api_key)

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
                f"{schema_label(response_model)} response did not match the schema."
            )
        return parsed, extract_usage(response)
