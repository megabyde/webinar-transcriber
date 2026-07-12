"""Provider configuration and env-driven construction for the optional cloud LLM."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from webinar_transcriber._env import llm_provider_name
from webinar_transcriber.llm.types import LlmConfigurationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from webinar_transcriber.llm.processor import InstructorLLMProcessor


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    label: str
    request_kwargs: Mapping[str, object] = field(default_factory=dict)


PROVIDERS = {
    "openai": ProviderSpec(label="OpenAI"),
    # Anthropic's Messages API requires max_tokens; OpenAI's is optional.
    "anthropic": ProviderSpec(label="Anthropic", request_kwargs={"max_tokens": 4096}),
}


def required_provider_env(provider_name: str) -> tuple[str, str]:
    """Return the required model name and API key for one provider.

    The env var names follow the SDK-standard ``{PROVIDER}_API_KEY`` / ``{PROVIDER}_MODEL`` pattern.

    Returns:
        tuple[str, str]: The configured model name and API key.

    Raises:
        LlmConfigurationError: If either required environment variable is missing.
    """
    prefix = provider_name.upper()
    api_key = os.environ.get(f"{prefix}_API_KEY")
    model_name = os.environ.get(f"{prefix}_MODEL")
    if not (model_name and api_key):
        raise LlmConfigurationError(
            f"Set {prefix}_API_KEY and {prefix}_MODEL for the '{provider_name}' LLM provider."
        )
    return model_name, api_key


def _required_llm_module(module_name: str, *, provider_label: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ImportError as ex:
        raise LlmConfigurationError(
            f"The {provider_label} provider requires the 'llm' extra. Install it with: "
            'uv tool install --reinstall ".[llm]"'
        ) from ex


def build_llm_processor_from_env(*, threads: int) -> InstructorLLMProcessor:
    """Build a configured LLM processor from environment variables.

    Returns:
        InstructorLLMProcessor: The configured provider-backed processor.

    Raises:
        LlmConfigurationError: If the provider selection or required environment is invalid.
    """
    # The CLI imports this package on base installs. Defer processor.py so its llm-extra-only
    # dependencies are loaded only when the user enables LLM processing.
    from webinar_transcriber.llm.processor import InstructorLLMProcessor  # noqa: PLC0415

    provider_name = llm_provider_name()
    spec = PROVIDERS.get(provider_name)
    if spec is None:
        raise LlmConfigurationError(
            "Unsupported LLM provider. Set LLM_PROVIDER to 'openai' or 'anthropic'."
        )

    instructor: Any = _required_llm_module("instructor", provider_label=spec.label)
    _required_llm_module(provider_name, provider_label=spec.label)
    model_name, api_key = required_provider_env(provider_name)

    return InstructorLLMProcessor(
        client=instructor.from_provider(
            f"{provider_name}/{model_name}", api_key=api_key, mode=instructor.Mode.TOOLS
        ),
        provider_name=provider_name,
        model_name=model_name,
        request_kwargs=spec.request_kwargs,
        threads=threads,
    )
