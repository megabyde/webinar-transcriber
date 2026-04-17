"""ASR pipeline helpers used by the top-level processor orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from webinar_transcriber.labels import count_label
from webinar_transcriber.models import InferenceWindow, TranscriptionResult
from webinar_transcriber.normalized_audio import load_normalized_audio
from webinar_transcriber.segmentation import (
    detect_speech_regions,
    normalized_audio_duration,
    repair_speech_regions,
)
from webinar_transcriber.transcript import reconcile_decoded_windows
from webinar_transcriber.transcript.normalize import normalize_transcription

from .support import (
    asr_runtime_detail,
    progress_stage,
    stage,
    window_transcription_stage_detail,
    write_json,
    write_model_json,
)

if TYPE_CHECKING:
    from pathlib import Path

    from webinar_transcriber.asr import WhisperCppTranscriber
    from webinar_transcriber.models import MediaAsset
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.segmentation import VadSettings

    from . import _AsrPipelineState
    from .support import ProgressStageHandle, StageContext


@dataclass(frozen=True)
class AsrPipelineResult:
    """Raw and normalized transcription outputs from the ASR pipeline."""

    transcription: TranscriptionResult
    normalized_transcription: TranscriptionResult


def run_asr_pipeline(
    *,
    audio_path: Path,
    media_asset: MediaAsset,
    transcriber: WhisperCppTranscriber,
    layout: RunLayout,
    ctx: StageContext,
    warnings: list[str],
    asr_pipeline: _AsrPipelineState,
    vad: VadSettings,
) -> AsrPipelineResult:
    """Run the deterministic local ASR pipeline and persist intermediate artifacts.

    Returns:
        AsrPipelineResult: The raw and normalized transcription outputs.
    """
    with stage(ctx, "prepare_asr", "Preparing ASR model") as st:
        transcriber.prepare_model()
        st.detail = asr_runtime_detail(transcriber)

    audio_samples, sample_rate = load_normalized_audio(audio_path)
    asr_pipeline.normalized_audio_duration_sec = normalized_audio_duration(
        audio_samples, sample_rate
    )
    with progress_stage(
        ctx,
        "vad",
        "Detecting speech regions",
        total=media_asset.duration_sec,
        count_label="s",
    ) as st:

        def on_vad_progress(
            completed_sec: float,
            detected_count: int,
            handle: ProgressStageHandle = st,
        ) -> None:
            handle.advance_to(completed_sec, detail=count_label(detected_count, "region"))

        speech_regions, vad_warnings = detect_speech_regions(
            audio_samples,
            sample_rate,
            enabled=vad.enabled,
            threshold=vad.threshold,
            min_speech_duration_ms=vad.min_speech_duration_ms,
            min_silence_duration_ms=vad.min_silence_duration_ms,
            speech_pad_ms=vad.speech_region_pad_ms,
            progress_callback=on_vad_progress,
        )
        st.finish_progress(
            media_asset.duration_sec, detail=count_label(len(speech_regions), "region")
        )
    asr_pipeline.vad_region_count = len(speech_regions)
    for warning in vad_warnings:
        warnings.append(warning)
        ctx.reporter.warn(warning)
    write_json(
        layout.speech_regions_path,
        {"speech_regions": [region.model_dump(mode="json") for region in speech_regions]},
    )

    with stage(ctx, "prepare_speech_regions", "Preparing speech regions") as st:
        expanded_regions = repair_speech_regions(speech_regions)
        windows = [
            InferenceWindow(
                window_id=f"window-{i + 1}",
                region_index=i,
                start_sec=region.start_sec,
                end_sec=region.end_sec,
            )
            for i, region in enumerate(expanded_regions)
            if region.end_sec > region.start_sec
        ]
        write_json(
            layout.expanded_regions_path,
            {"expanded_regions": [region.model_dump(mode="json") for region in expanded_regions]},
        )
        st.detail = (
            f"{count_label(len(speech_regions), 'region')} "
            f"-> {count_label(len(expanded_regions), 'region')} "
            f"| {count_label(len(windows), 'window')}"
        )
    asr_pipeline.window_count = len(windows)
    asr_pipeline.average_window_duration_sec = (
        sum(w.end_sec - w.start_sec for w in windows) / len(windows) if windows else None
    )
    with progress_stage(
        ctx,
        "transcribe",
        "Transcribing audio",
        total=media_asset.duration_sec,
        count_label="s",
        detail="0 segments",
    ) as st:

        def on_window_completed(
            completed_sec: float,
            segment_count: int,
            handle: ProgressStageHandle = st,
        ) -> None:
            handle.advance_to(completed_sec, detail=count_label(segment_count, "segment"))

        decoded_windows = transcriber.transcribe_inference_windows(
            audio_samples,
            windows,
            progress_callback=on_window_completed,
        )
        write_json(
            layout.decoded_windows_path,
            {"decoded_windows": [window.model_dump(mode="json") for window in decoded_windows]},
        )
        st.finish_progress(media_asset.duration_sec)
        st.detail = window_transcription_stage_detail(
            window_count=len(windows),
            total_duration_sec=media_asset.duration_sec,
            elapsed_sec=st.elapsed_sec(),
        )

    with stage(ctx, "reconcile", "Reconciling transcript windows") as st:
        transcription, reconciliation_stats = reconcile_decoded_windows(decoded_windows)
        st.detail = count_label(len(transcription.segments), "segment")
    asr_pipeline.reconciliation_duplicate_segments_dropped = (
        reconciliation_stats.duplicate_segments_dropped
    )
    asr_pipeline.reconciliation_boundary_fixes = reconciliation_stats.boundary_fixes
    asr_pipeline.system_info = transcriber.system_info

    write_model_json(layout.transcript_path, transcription)
    normalized_transcription = normalize_transcription(transcription)
    return AsrPipelineResult(
        transcription=transcription, normalized_transcription=normalized_transcription
    )
