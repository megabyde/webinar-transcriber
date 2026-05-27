"""High-level processing orchestration."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
from typing import TYPE_CHECKING

from webinar_transcriber.asr import WhisperCppTranscriber
from webinar_transcriber.diagnostics import write_run_diagnostics
from webinar_transcriber.media import probe_media
from webinar_transcriber.normalized_audio import (
    prepared_transcription_audio,
    preserve_transcription_audio,
)
from webinar_transcriber.paths import create_run_layout
from webinar_transcriber.reporter import BaseStageReporter

from .asr_pipeline import AsrPipelineResult, run_asr_pipeline
from .report import run_report_phase
from .support import progress_stage, stage, write_json
from .types import (
    ProcessArtifacts,
    RunContext,
    TranscriptionConfig,
)

__all__ = [
    "ProcessArtifacts",
    "RunContext",
    "TranscriptionConfig",
    "process_input",
    "run_transcription_phase",
]

if TYPE_CHECKING:
    from pathlib import Path

    from webinar_transcriber.diarization import Diarizer
    from webinar_transcriber.llm.contracts import LLMProcessor
    from webinar_transcriber.models import (
        MediaAsset,
        ReportDocument,
        Scene,
        SceneFrame,
    )
    from webinar_transcriber.paths import RunLayout


def run_transcription_phase(
    *,
    input_path: Path,
    media_asset: MediaAsset,
    transcriber: WhisperCppTranscriber,
    layout: RunLayout,
    ctx: RunContext,
    transcription_config: TranscriptionConfig,
    diarizer: Diarizer | None = None,
    diarization_speaker_count: int | None = None,
) -> AsrPipelineResult:
    """Run the audio preparation and ASR half of the pipeline.

    Returns:
        AsrPipelineResult: The transcription artifacts and ASR diagnostics.
    """
    with ExitStack() as audio_scope:
        transcription_audio_total = max(media_asset.duration_sec, 1.0)
        with progress_stage(
            ctx,
            "prepare_transcription_audio",
            "Preparing audio",
            total=transcription_audio_total,
            count_label="s",
        ) as st:
            audio_path = audio_scope.enter_context(
                prepared_transcription_audio(
                    input_path,
                    progress_callback=lambda completed_sec: st.advance_to(
                        min(completed_sec, transcription_audio_total)
                    ),
                )
            )
            st.advance_to(transcription_audio_total, detail=audio_path.name)
            st.set_detail(str(audio_path.name))

        asr_result = run_asr_pipeline(
            audio_path=audio_path,
            media_asset=media_asset,
            transcriber=transcriber,
            layout=layout,
            ctx=ctx,
            transcription_config=transcription_config,
            diarizer=diarizer,
            diarization_speaker_count=diarization_speaker_count,
        )

        if transcription_config.keep_audio:
            with progress_stage(
                ctx,
                "save_transcription_audio",
                "Saving transcription audio",
                total=transcription_audio_total,
                count_label="s",
            ) as st:
                preserved_audio_path = layout.transcription_audio_path()
                preserve_transcription_audio(
                    audio_path,
                    preserved_audio_path,
                    progress_callback=lambda completed_sec: st.advance_to(
                        min(completed_sec, transcription_audio_total)
                    ),
                )
                st.advance_to(transcription_audio_total, detail=preserved_audio_path.name)
                st.set_detail(preserved_audio_path.name)

    return asr_result


def process_input(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    transcription_config: TranscriptionConfig,
    llm_processor: LLMProcessor | None = None,
    diarizer: Diarizer | None = None,
    diarization_speaker_count: int | None = None,
    transcriber: WhisperCppTranscriber | None = None,
    reporter: BaseStageReporter | None = None,
) -> ProcessArtifacts:
    """Process a single audio or video file into report artifacts.

    Returns:
        ProcessArtifacts: The completed processing artifacts.
    """
    active_reporter = reporter or BaseStageReporter()
    ctx = RunContext(reporter=active_reporter)

    transcriber = transcriber or WhisperCppTranscriber(
        model_name=transcription_config.asr_model,
        threads=transcription_config.threads,
        language=transcription_config.language,
    )

    with transcriber as active_transcriber:
        asr_result: AsrPipelineResult | None = None
        report: ReportDocument | None = None
        scenes: list[Scene] = []
        scene_frames: list[SceneFrame] = []
        ctx.reporter.begin_run(input_path)
        try:
            with stage(ctx, "prepare_run_dir", "Preparing run directory") as st:
                layout = create_run_layout(input_path=input_path, output_dir=output_dir)
                ctx.layout = layout
                st.set_detail(str(layout.run_dir))
            active_transcriber.set_log_path(layout.run_dir / "whisper-cpp.log")

            with stage(ctx, "probe_media", "Probing media") as st:
                media_asset = probe_media(input_path)
                write_json(layout.metadata_path, asdict(media_asset))
                st.set_detail(f"{media_asset.media_type.value}, {media_asset.duration_sec:.1f}s")

            asr_result = run_transcription_phase(
                input_path=input_path,
                media_asset=media_asset,
                transcriber=active_transcriber,
                layout=layout,
                ctx=ctx,
                transcription_config=transcription_config,
                diarizer=diarizer,
                diarization_speaker_count=diarization_speaker_count,
            )

            report, scenes, scene_frames = run_report_phase(
                input_path=input_path,
                layout=layout,
                media_asset=media_asset,
                normalized_transcription=asr_result.normalized_transcription,
                llm_processor=llm_processor,
                ctx=ctx,
            )

            diagnostics = write_run_diagnostics(
                ctx,
                status="succeeded",
                llm_enabled=llm_processor is not None,
                transcription=asr_result.transcription,
                normalized_transcription=asr_result.normalized_transcription,
                asr_pipeline=asr_result.asr_pipeline,
                diarization=asr_result.diarization,
                report=report,
                scenes=scenes,
                scene_frames=scene_frames,
            )
            if diagnostics is None:  # pragma: no cover - run layout always exists on success
                raise RuntimeError(
                    "Run diagnostics were not written after creating the run layout."
                )

            artifacts = ProcessArtifacts(
                layout=layout,
                media_asset=media_asset,
                transcription=asr_result.transcription,
                report=report,
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
                llm_enabled=llm_processor is not None,
                transcription=asr_result.transcription if asr_result else None,
                normalized_transcription=(
                    asr_result.normalized_transcription if asr_result else None
                ),
                asr_pipeline=asr_result.asr_pipeline if asr_result else None,
                diarization=asr_result.diarization if asr_result else None,
                report=report,
                scenes=scenes,
                scene_frames=scene_frames,
                suppress_errors=True,
            )
            raise
