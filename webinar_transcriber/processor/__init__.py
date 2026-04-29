"""High-level processing orchestration."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from typing import TYPE_CHECKING

import webinar_transcriber.asr as asr_runtime
import webinar_transcriber.media as media_runtime
from webinar_transcriber.asr import PromptCarryoverSettings, default_asr_threads
from webinar_transcriber.diagnostics import write_run_diagnostics
from webinar_transcriber.normalized_audio import (
    prepared_transcription_audio,
    preserve_transcription_audio,
)
from webinar_transcriber.paths import create_run_layout
from webinar_transcriber.reporter import BaseStageReporter
from webinar_transcriber.segmentation import VadSettings

from . import asr as processor_asr
from .report import run_report_phase
from .support import stage, write_json
from .types import AsrPipelineState, ProcessArtifacts, RunContext, TranscriptionPhaseResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from pathlib import Path

    from webinar_transcriber.asr import WhisperCppTranscriber
    from webinar_transcriber.llm import LLMProcessor
    from webinar_transcriber.models import MediaAsset
    from webinar_transcriber.paths import RunLayout


DEFAULT_VAD_SETTINGS = VadSettings()
DEFAULT_PROMPT_CARRYOVER_SETTINGS = PromptCarryoverSettings()


def run_transcription_phase(
    *,
    input_path: Path,
    media_asset: MediaAsset,
    transcriber: WhisperCppTranscriber,
    layout: RunLayout,
    ctx: RunContext,
    vad: VadSettings,
    carryover_enabled: bool,
    language: str | None,
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
            language=language,
        )

        if keep_audio:
            with stage(ctx, "save_transcription_audio", "Saving transcription audio") as st:
                preserved_audio_path = layout.transcription_audio_path(kept_audio_format)
                preserve_transcription_audio(
                    audio_path, preserved_audio_path, audio_format=kept_audio_format
                )
                st.detail = preserved_audio_path.name

    return TranscriptionPhaseResult(
        transcription=asr_result.transcription,
        normalized_transcription=asr_result.normalized_transcription,
        asr_pipeline=asr_result.asr_pipeline,
    )


def process_input(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    asr_model: str | None = None,
    language: str | None = None,
    vad: VadSettings = DEFAULT_VAD_SETTINGS,
    carryover: PromptCarryoverSettings = DEFAULT_PROMPT_CARRYOVER_SETTINGS,
    asr_threads: int | None = None,
    keep_audio: bool = False,
    kept_audio_format: str = "wav",
    enable_llm: bool = False,
    transcriber: WhisperCppTranscriber | None = None,
    llm_processor: LLMProcessor | None = None,
    reporter: BaseStageReporter | None = None,
) -> ProcessArtifacts:
    """Process a single audio or video file into report artifacts.

    Returns:
        ProcessArtifacts: The completed processing artifacts.
    """
    active_reporter = reporter or BaseStageReporter()
    asr_threads = asr_threads or default_asr_threads()
    ctx = RunContext(
        reporter=active_reporter,
        asr_pipeline=AsrPipelineState(
            vad_enabled=vad.enabled, threads=asr_threads, carryover_enabled=carryover.enabled
        ),
    )

    active_transcriber = transcriber or asr_runtime.WhisperCppTranscriber(
        model_name=asr_model, threads=asr_threads, language=language, carryover_settings=carryover
    )
    transcriber_scope = (
        active_transcriber if transcriber is None else nullcontext(active_transcriber)
    )
    with transcriber_scope as active_transcriber:
        ctx.reporter.begin_run(input_path)
        try:
            with stage(ctx, "prepare_run_dir", "Preparing run directory") as st:
                layout = create_run_layout(input_path=input_path, output_dir=output_dir)
                ctx.layout = layout
                st.detail = str(layout.run_dir)
            active_transcriber.set_log_path(layout.run_dir / "whisper-cpp.log")

            with stage(ctx, "probe_media", "Probing media") as st:
                media_asset = media_runtime.probe_media(input_path)
                ctx.media_asset = media_asset
                write_json(layout.metadata_path, {"media": asdict(media_asset)})
                st.detail = f"{media_asset.media_type.value}, {media_asset.duration_sec:.1f}s"

            transcription_phase = run_transcription_phase(
                input_path=input_path,
                media_asset=media_asset,
                transcriber=active_transcriber,
                layout=layout,
                ctx=ctx,
                vad=vad,
                carryover_enabled=carryover.enabled,
                language=language,
                keep_audio=keep_audio,
                kept_audio_format=kept_audio_format,
                prepared_audio_factory=prepared_transcription_audio,
            )
            ctx.transcription = transcription_phase.transcription
            ctx.normalized_transcription = transcription_phase.normalized_transcription
            ctx.asr_pipeline = transcription_phase.asr_pipeline

            report_phase = run_report_phase(
                input_path=input_path,
                layout=layout,
                media_asset=media_asset,
                normalized_transcription=transcription_phase.normalized_transcription,
                enable_llm=enable_llm,
                llm_processor=llm_processor,
                ctx=ctx,
            )
            ctx.alignment_blocks = report_phase.alignment_blocks
            ctx.scenes = report_phase.scenes
            ctx.slide_frames = report_phase.slide_frames
            ctx.report = report_phase.report

            diagnostics = write_run_diagnostics(
                ctx,
                status="succeeded",
                asr_model=active_transcriber.model_name,
                llm_enabled=enable_llm,
            )
            if diagnostics is None:  # pragma: no cover - run layout always exists on success
                raise RuntimeError(
                    "Run diagnostics were not written after creating the run layout."
                )

            artifacts = ProcessArtifacts(
                layout=layout,
                media_asset=media_asset,
                transcription=transcription_phase.transcription,
                report=report_phase.report,
                diagnostics=diagnostics,
            )
            ctx.reporter.complete_run(artifacts)
            return artifacts
        except Exception as ex:
            write_run_diagnostics(
                ctx,
                status="failed",
                failed_stage=ctx.current_stage,
                error=str(ex),
                asr_model=active_transcriber.model_name,
                llm_enabled=enable_llm,
                suppress_errors=True,
            )
            raise
