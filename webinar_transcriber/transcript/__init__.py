"""Transcript-domain normalization and reconciliation helpers."""

from __future__ import annotations

from webinar_transcriber.transcript.coalesce import coalesce_transcript
from webinar_transcriber.transcript.reconcile import reconcile_decoded_windows

__all__ = ["coalesce_transcript", "reconcile_decoded_windows"]
