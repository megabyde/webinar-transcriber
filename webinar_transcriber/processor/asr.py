"""ASR pipeline helpers used by the top-level processor orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

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
    count_label,
    progress_stage,
    stage,
    window_transcription_stage_detail,
    write_json,
    write_model_json,
)
from .types import AsrPipelineState

if TYPE_CHECKING:
    from pathlib import Path

    from webinar_transcriber.asr import WhisperCppTranscriber
    from webinar_transcriber.models import MediaAsset
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.segmentation import VadSettings

    from .support import ProgressStageHandle
    from .types import RunContext


@dataclass(frozen=True)
class AsrPipelineResult:
    """Raw and normalized transcription outputs from the ASR pipeline."""

    transcription: TranscriptionResult
    normalized_transcription: TranscriptionResult
    asr_pipeline: AsrPipelineState


def run_asr_pipeline(
    *,
    audio_path: Path,
    media_asset: MediaAsset,
    transcriber: WhisperCppTranscriber,
    layout: RunLayout,
    ctx: RunContext,
    warnings: list[str],
    vad: VadSettings,
    carryover_enabled: bool,
) -> AsrPipelineResult:
    """Run the deterministic local ASR pipeline and persist intermediate artifacts.

    Returns:
        AsrPipelineResult: The raw and normalized transcription outputs.
    """
    with stage(ctx, "prepare_asr", "Preparing ASR model") as st:
        transcriber.prepare_model()
        st.detail = asr_runtime_detail(transcriber)

    audio_samples, sample_rate = load_normalized_audio(audio_path)
    normalized_audio_duration_sec = normalized_audio_duration(audio_samples, sample_rate)
    with stage(ctx, "vad", "Detecting speech regions") as st:
        speech_regions, vad_warnings = detect_speech_regions(
            audio_samples,
            sample_rate,
            enabled=vad.enabled,
            threshold=vad.threshold,
            min_speech_duration_ms=vad.min_speech_duration_ms,
            min_silence_duration_ms=vad.min_silence_duration_ms,
            speech_pad_ms=vad.speech_region_pad_ms,
            progress_callback=None,
        )
        st.detail = count_label(len(speech_regions), "region")
    vad_region_count = len(speech_regions)
    for warning in vad_warnings:
        warnings.append(warning)
        ctx.reporter.warn(warning)
    write_json(
        layout.speech_regions_path,
        {"speech_regions": [asdict(region) for region in speech_regions]},
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
            {"expanded_regions": [asdict(region) for region in expanded_regions]},
        )
        st.detail = count_label(len(expanded_regions), "region")
    window_count = len(windows)
    average_window_duration_sec = (
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
            {"decoded_windows": [asdict(window) for window in decoded_windows]},
        )
        st.finish_progress(media_asset.duration_sec)
        st.detail = window_transcription_stage_detail(
            window_count=len(windows),
            total_duration_sec=media_asset.duration_sec,
            elapsed_sec=st.elapsed_sec(),
        )

    with stage(ctx, "reconcile", "Reconciling transcript windows") as st:
        decoded_segment_count = sum(len(window.segments) for window in decoded_windows)
        transcription, reconciliation_stats = reconcile_decoded_windows(decoded_windows)
        segment_label = "segment" if len(transcription.segments) == 1 else "segments"
        st.detail = f"{decoded_segment_count} -> {len(transcription.segments)} {segment_label}"

    with stage(ctx, "normalize_transcript", "Normalizing transcript") as st:
        write_model_json(layout.transcript_path, transcription)
        normalized_transcription = normalize_transcription(transcription)
        st.detail = count_label(len(normalized_transcription.segments), "segment")

    asr_pipeline = AsrPipelineState(
        vad_enabled=vad.enabled,
        threads=transcriber.threads,
        normalized_audio_duration_sec=normalized_audio_duration_sec,
        vad_region_count=vad_region_count,
        carryover_enabled=carryover_enabled,
        window_count=window_count,
        average_window_duration_sec=average_window_duration_sec,
        reconciliation_boundary_fixes=reconciliation_stats.boundary_fixes,
        system_info=transcriber.system_info,
    )
    return AsrPipelineResult(
        transcription=transcription,
        normalized_transcription=normalized_transcription,
        asr_pipeline=asr_pipeline,
    )
