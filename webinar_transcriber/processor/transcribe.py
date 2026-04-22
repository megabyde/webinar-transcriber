"""Private transcription-phase orchestration helpers."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

from webinar_transcriber.normalized_audio import preserve_transcription_audio

from . import asr as processor_asr
from .support import stage
from .types import TranscriptionPhaseResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from pathlib import Path

    from webinar_transcriber.asr import WhisperCppTranscriber
    from webinar_transcriber.models import MediaAsset
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.segmentation import VadSettings

    from .types import RunContext


def run_transcription_phase(
    *,
    input_path: Path,
    media_asset: MediaAsset,
    transcriber: WhisperCppTranscriber,
    layout: RunLayout,
    ctx: RunContext,
    vad: VadSettings,
    carryover_enabled: bool,
    keep_audio: bool,
    kept_audio_format: str,
    prepared_audio_factory: Callable[[Path], AbstractContextManager[Path]],
) -> TranscriptionPhaseResult:
    """Run the audio preparation and ASR half of the pipeline.

    Returns:
        TranscriptionPhaseResult: The transcription artifacts and ASR diagnostics.
    """
    with ExitStack() as audio_scope:
        with stage(ctx, "prepare_transcription_audio", "Preparing audio") as st:
            audio_path = audio_scope.enter_context(prepared_audio_factory(input_path))
            st.detail = str(audio_path.name)

        asr_result = processor_asr.run_asr_pipeline(
            audio_path=audio_path,
            media_asset=media_asset,
            transcriber=transcriber,
            layout=layout,
            ctx=ctx,
            warnings=ctx.warnings,
            vad=vad,
            carryover_enabled=carryover_enabled,
        )

        if keep_audio:
            preserved_audio_path = layout.transcription_audio_path(kept_audio_format)
            preserve_transcription_audio(
                audio_path,
                preserved_audio_path,
                audio_format=kept_audio_format,
            )

    return TranscriptionPhaseResult(
        transcription=asr_result.transcription,
        normalized_transcription=asr_result.normalized_transcription,
        asr_pipeline=asr_result.asr_pipeline,
    )
