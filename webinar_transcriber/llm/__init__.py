"""Optional cloud LLM helpers for transcript and report enhancement."""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Protocol, cast

from .contracts import (
    LLMConfigurationError,
    LLMProcessingError,
    LLMProcessor,
    LLMReportMetadataResult,
    LLMReportPolishPlan,
    LLMSectionPolishResult,
    ReportPolishResponse,
    ReportSectionUpdate,
    SectionTextResponse,
)
from .processor import InstructorLLMProcessor
from .prompts import ACTION_ITEM_LIMIT, REPORT_POLISH_TOTAL_CHAR_BUDGET, SECTION_POLISH_MAX_WORKERS
from .utils import required_provider_env

if TYPE_CHECKING:
    from .processor import InstructorClient

    class _InstructorMode(Protocol):
        TOOLS: object

    class InstructorModule(Protocol):
        Mode: _InstructorMode

        def from_provider(self, provider_model: str, **kwargs: object) -> InstructorClient: ...


def _required_llm_module(module_name: str, *, provider_label: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise LLMConfigurationError(
            f"The {provider_label} provider requires the 'llm' extra. "
            'Install it with: uv tool install --reinstall ".[llm]"'
        ) from error


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
            instructor = cast(
                "InstructorModule", _required_llm_module("instructor", provider_label="OpenAI")
            )
            _required_llm_module("openai", provider_label="OpenAI")
            api_key, model_name = required_provider_env(
                api_key_env="OPENAI_API_KEY", model_env="OPENAI_MODEL"
            )
            return InstructorLLMProcessor(
                client=instructor.from_provider(
                    f"openai/{model_name}", api_key=api_key, mode=instructor.Mode.TOOLS
                ),
                provider_name="openai",
                model_name=model_name,
            )
        case "anthropic":
            instructor = cast(
                "InstructorModule", _required_llm_module("instructor", provider_label="Anthropic")
            )
            _required_llm_module("anthropic", provider_label="Anthropic")
            api_key, model_name = required_provider_env(
                api_key_env="ANTHROPIC_API_KEY", model_env="ANTHROPIC_MODEL"
            )
            return InstructorLLMProcessor(
                client=instructor.from_provider(
                    f"anthropic/{model_name}", api_key=api_key, mode=instructor.Mode.TOOLS
                ),
                provider_name="anthropic",
                model_name=model_name,
                request_kwargs={"max_tokens": 4096},
            )
        case _:
            raise LLMConfigurationError(
                "Unsupported LLM provider. Set LLM_PROVIDER to 'openai' or 'anthropic'."
            )


__all__ = [
    "ACTION_ITEM_LIMIT",
    "REPORT_POLISH_TOTAL_CHAR_BUDGET",
    "SECTION_POLISH_MAX_WORKERS",
    "InstructorLLMProcessor",
    "LLMConfigurationError",
    "LLMProcessingError",
    "LLMProcessor",
    "LLMReportMetadataResult",
    "LLMReportPolishPlan",
    "LLMSectionPolishResult",
    "ReportPolishResponse",
    "ReportSectionUpdate",
    "SectionTextResponse",
    "build_llm_processor_from_env",
]
