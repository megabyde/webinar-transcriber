"""Optional cloud LLM helpers for transcript and report enhancement."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from webinar_transcriber._env import llm_provider_name

from .contracts import (
    LLMConfigurationError,
    LLMProcessingError,
    LLMProcessor,
    LLMReportMetadataResult,
    LLMReportPolishPlan,
    LLMSectionPolishResult,
)
from .prompts import ACTION_ITEM_LIMIT, REPORT_POLISH_TOTAL_CHAR_BUDGET
from .utils import required_provider_env

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .processor import InstructorLLMProcessor
    from .schemas import ReportPolishResponse, ReportSectionUpdate, SectionTextResponse


_EMPTY_REQUEST_KWARGS = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    label: str
    module_name: str
    api_key_env: str
    model_env: str
    request_kwargs: Mapping[str, object] = _EMPTY_REQUEST_KWARGS


PROVIDERS = {
    "openai": ProviderSpec(
        label="OpenAI",
        module_name="openai",
        api_key_env="OPENAI_API_KEY",
        model_env="OPENAI_MODEL",
    ),
    "anthropic": ProviderSpec(
        label="Anthropic",
        module_name="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        model_env="ANTHROPIC_MODEL",
        request_kwargs={"max_tokens": 4096},
    ),
}


def _required_llm_module(module_name: str, *, provider_label: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise LLMConfigurationError(
            f"The {provider_label} provider requires the 'llm' extra. "
            'Install it with: uv tool install --reinstall ".[llm]"'
        ) from error


def ensure_llm_extra_available() -> None:
    """Raise when the configured provider cannot load the optional LLM dependencies."""
    provider = llm_provider_name()
    spec = PROVIDERS.get(provider)
    if spec is None:
        return
    _required_llm_module("instructor", provider_label=spec.label)
    _required_llm_module(spec.module_name, provider_label=spec.label)


def build_llm_processor_from_env(*, threads: int) -> LLMProcessor:
    """Build a configured LLM processor from environment variables.

    Returns:
        LLMProcessor: The configured provider-backed processor.

    Raises:
        LLMConfigurationError: If the provider selection or required environment is invalid.
    """
    from .processor import InstructorLLMProcessor  # noqa: PLC0415

    provider_name = llm_provider_name()
    spec = PROVIDERS.get(provider_name)
    if spec is None:
        raise LLMConfigurationError(
            "Unsupported LLM provider. Set LLM_PROVIDER to 'openai' or 'anthropic'."
        )

    instructor = cast("Any", _required_llm_module("instructor", provider_label=spec.label))
    _required_llm_module(spec.module_name, provider_label=spec.label)
    api_key, model_name = required_provider_env(
        api_key_env=spec.api_key_env, model_env=spec.model_env
    )

    return InstructorLLMProcessor(
        client=instructor.from_provider(
            f"{provider_name}/{model_name}", api_key=api_key, mode=instructor.Mode.TOOLS
        ),
        provider_name=provider_name,
        model_name=model_name,
        request_kwargs=spec.request_kwargs,
        threads=threads,
    )


_LAZY_EXPORTS = {
    "InstructorLLMProcessor": ("webinar_transcriber.llm.processor", "InstructorLLMProcessor"),
    "ReportPolishResponse": ("webinar_transcriber.llm.schemas", "ReportPolishResponse"),
    "ReportSectionUpdate": ("webinar_transcriber.llm.schemas", "ReportSectionUpdate"),
    "SectionTextResponse": ("webinar_transcriber.llm.schemas", "SectionTextResponse"),
}


def __getattr__(name: str) -> object:
    """Import provider-specific LLM helpers only when requested."""
    if name not in _LAZY_EXPORTS:  # pragma: no cover - Python module protocol fallback
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    return getattr(importlib.import_module(module_name), attribute_name)


__all__ = [
    "ACTION_ITEM_LIMIT",
    "REPORT_POLISH_TOTAL_CHAR_BUDGET",
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
    "ensure_llm_extra_available",
]
