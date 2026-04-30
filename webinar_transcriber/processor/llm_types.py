"""Provider-neutral LLM runtime state used by processor orchestration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMRuntimeState:
    """Observed state for the optional LLM report stage."""

    provider_name: str | None = None
    model_name: str | None = None
    report_status: str = "disabled"
    report_latency_sec: float | None = None
    report_usage: dict[str, int] | None = None
