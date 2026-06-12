"""Optional cloud LLM helpers for transcript and report enhancement."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from webinar_transcriber._env import llm_provider_name

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .processor import InstructorLLMProcessor


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class LlmConfigurationError(RuntimeError):
    """Raised when required LLM configuration is missing."""


class LlmProcessingError(RuntimeError):
    """Raised when the LLM response cannot be validated or applied."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LlmSectionPolishResult:
    """Validated result from the section-text polishing phase."""

    section_tldrs: dict[str, str]
    section_transcripts: dict[str, str]
    response_metadata: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LlmReportMetadataResult:
    """Validated result from the final report metadata polishing phase."""

    summary: list[str]
    action_items: list[str]
    section_titles: dict[str, str]
    response_metadata: list[dict[str, object]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    label: str
    module_name: str
    api_key_env: str
    model_env: str
    request_kwargs: Mapping[str, object] = field(default_factory=dict)


PROVIDERS = {
    "openai": ProviderSpec(
        label="OpenAI", module_name="openai", api_key_env="OPENAI_API_KEY", model_env="OPENAI_MODEL"
    ),
    "anthropic": ProviderSpec(
        label="Anthropic",
        module_name="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        model_env="ANTHROPIC_MODEL",
        request_kwargs={"max_tokens": 4096},
    ),
}


def required_provider_env(*, api_key_env: str, model_env: str) -> tuple[str, str]:
    """Return the required API key and model name for one provider.

    Returns:
        tuple[str, str]: The configured API key and model name.

    Raises:
        LlmConfigurationError: If either required environment variable is missing.
    """
    api_key = os.environ.get(api_key_env)
    model_name = os.environ.get(model_env)
    if not api_key or not model_name:
        missing = ", ".join(
            env_name
            for env_name, value in ((api_key_env, api_key), (model_env, model_name))
            if not value
        )
        raise LlmConfigurationError(f"Missing required LLM environment variables: {missing}.")
    return api_key, model_name


def _required_llm_module(module_name: str, *, provider_label: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise LlmConfigurationError(
            f"The {provider_label} provider requires the 'llm' extra. Install it with: "
            'uv tool install --reinstall ".[llm]"'
        ) from error


def build_llm_processor_from_env(*, threads: int) -> InstructorLLMProcessor:
    """Build a configured LLM processor from environment variables.

    Returns:
        InstructorLLMProcessor: The configured provider-backed processor.

    Raises:
        LlmConfigurationError: If the provider selection or required environment is invalid.
    """
    from .processor import InstructorLLMProcessor  # noqa: PLC0415

    provider_name = llm_provider_name()
    spec = PROVIDERS.get(provider_name)
    if spec is None:
        raise LlmConfigurationError(
            "Unsupported LLM provider. Set LLM_PROVIDER to 'openai' or 'anthropic'."
        )

    instructor: Any = _required_llm_module("instructor", provider_label=spec.label)
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


__all__ = [
    "LlmConfigurationError",
    "LlmProcessingError",
    "LlmReportMetadataResult",
    "LlmSectionPolishResult",
    "build_llm_processor_from_env",
]
