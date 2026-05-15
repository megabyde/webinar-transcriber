"""Provider-neutral LLM runtime state used by processor orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from webinar_transcriber.models import ReportStatus, TokenUsage


@dataclass
class LLMRuntimeState:
    """Observed state for the optional LLM report stage."""

    provider_name: str | None = None
    model_name: str | None = None
    report_status: ReportStatus = "disabled"
    report_latency_sec: float | None = None
    report_usage: TokenUsage | None = None
    response_metadata: list[dict[str, object]] = field(default_factory=list)
