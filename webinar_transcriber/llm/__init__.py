"""Optional cloud LLM helpers for transcript and report enhancement."""

from __future__ import annotations

import importlib
import os

from .anthropic_backend import AnthropicLLMProcessor
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
    SectionTextResponse,
)
from .openai_backend import OpenAILLMProcessor
from .prompts import (
    ACTION_ITEM_LIMIT,
    REPORT_POLISH_TOTAL_CHAR_BUDGET,
    SECTION_POLISH_MAX_WORKERS,
)
from .utils import required_provider_env


def build_llm_processor_from_env() -> LLMProcessor:
    """Build a configured LLM processor from environment variables.

    Returns:
        LLMProcessor: The configured provider-backed processor.

    Raises:
        LLMConfigurationError: If the provider selection or required environment is invalid.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").strip().casefold()
    match provider:
        case "openai":
            try:
                importlib.import_module("openai")
            except ImportError as error:
                raise LLMConfigurationError(
                    "The OpenAI provider requires the 'llm' extra. "
                    'Install it with: uv tool install --reinstall ".[llm]"'
                ) from error
            api_key, model_name = required_provider_env(
                api_key_env="OPENAI_API_KEY", model_env="OPENAI_MODEL"
            )
            return OpenAILLMProcessor(api_key=api_key, model_name=model_name)
        case "anthropic":
            try:
                importlib.import_module("anthropic")
            except ImportError as error:
                raise LLMConfigurationError(
                    "The Anthropic provider requires the 'llm' extra. "
                    'Install it with: uv tool install --reinstall ".[llm]"'
                ) from error
            api_key, model_name = required_provider_env(
                api_key_env="ANTHROPIC_API_KEY", model_env="ANTHROPIC_MODEL"
            )
            return AnthropicLLMProcessor(api_key=api_key, model_name=model_name)
        case _:
            raise LLMConfigurationError(
                "Unsupported LLM provider. Set LLM_PROVIDER to 'openai' or 'anthropic'."
            )


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
