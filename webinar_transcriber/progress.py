"""Shared progress callback contracts."""

from __future__ import annotations

from typing import Protocol


class ProgressCallback(Protocol):
    """Progress callback carrying completed work and a current item count."""

    def __call__(self, completed: float, count: int, /) -> None:
        """Report completed work and the current count for stage detail."""
