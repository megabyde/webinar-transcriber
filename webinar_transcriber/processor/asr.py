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
    progress_updater,
    start_stage_timer,
    window_transcription_stage_detail,
    write_json,
)

if TYPE_CHECKING:
    from pathlib import Path

    from webinar_transcriber.asr import WhisperCppTranscriber
    from webinar_transcriber.models import MediaAsset
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.reporter import StageReporter
    from webinar_transcriber.segmentation import VadSettings

    from . import _AsrPipelineState


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
    reporter: StageReporter,
    stage_timings: dict[str, float],
    warnings: list[str],
    asr_pipeline: _AsrPipelineState,
    vad: VadSettings,
) -> AsrPipelineResult:
    """Run the deterministic local ASR pipeline and persist intermediate artifacts."""
    reporter.stage_started("prepare_asr", "Preparing ASR model")
    timer = start_stage_timer(stage_timings, "prepare_asr")
    transcriber.prepare_model()
    timer.finish()
    reporter.stage_finished(
        "prepare_asr", "Preparing ASR model", detail=asr_runtime_detail(transcriber)
    )

    audio_samples, sample_rate = load_normalized_audio(audio_path)
    asr_pipeline.normalized_audio_duration_sec = normalized_audio_duration(
        audio_samples, sample_rate
    )
    reporter.progress_started(
        "vad", "Detecting speech regions", total=media_asset.duration_sec, count_label="s"
    )
    on_vad_progress, finish_vad_progress = progress_updater(reporter, stage_key="vad")

    timer = start_stage_timer(stage_timings, "vad")
    speech_regions, vad_warnings = detect_speech_regions(
        audio_samples,
        sample_rate,
        enabled=vad.enabled,
        threshold=vad.threshold,
        min_speech_duration_ms=vad.min_speech_duration_ms,
        min_silence_duration_ms=vad.min_silence_duration_ms,
        speech_pad_ms=vad.speech_region_pad_ms,
        progress_callback=lambda completed_sec, detected_count: on_vad_progress(
            completed_sec, detail=count_label(detected_count, "region")
        ),
    )
    finish_vad_progress(media_asset.duration_sec, detail=count_label(len(speech_regions), "region"))
    timer.finish()
    asr_pipeline.vad_region_count = len(speech_regions)
    reporter.stage_finished(
        "vad", "Detecting speech regions", detail=count_label(len(speech_regions), "region")
    )
    for warning in vad_warnings:
        warnings.append(warning)
        reporter.warn(warning)
    write_json(
        layout.speech_regions_path,
        {"speech_regions": [region.model_dump(mode="json") for region in speech_regions]},
    )

    reporter.stage_started("prepare_speech_regions", "Preparing speech regions")
    timer = start_stage_timer(stage_timings, "prepare_speech_regions")
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
    timer.finish()
    asr_pipeline.window_count = len(windows)
    asr_pipeline.average_window_duration_sec = (
        sum(w.end_sec - w.start_sec for w in windows) / len(windows) if windows else None
    )
    reporter.stage_finished(
        "prepare_speech_regions",
        "Preparing speech regions",
        detail=(
            f"{count_label(len(speech_regions), 'region')} "
            f"-> {count_label(len(expanded_regions), 'region')} "
            f"| {count_label(len(windows), 'window')}"
        ),
    )

    reporter.progress_started(
        "transcribe",
        "Transcribing audio",
        total=media_asset.duration_sec,
        count_label="s",
        detail="0 segments",
    )
    on_window_completed, finish_transcribe_progress = progress_updater(
        reporter, stage_key="transcribe"
    )

    timer = start_stage_timer(stage_timings, "transcribe")
    decoded_windows = transcriber.transcribe_inference_windows(
        audio_samples,
        windows,
        progress_callback=lambda completed_sec, segment_count: on_window_completed(
            completed_sec, detail=count_label(segment_count, "segment")
        ),
    )
    write_json(
        layout.decoded_windows_path,
        {"decoded_windows": [window.model_dump(mode="json") for window in decoded_windows]},
    )
    finish_transcribe_progress(media_asset.duration_sec)
    transcribe_elapsed_sec = timer.finish()
    reporter.stage_finished(
        "transcribe",
        "Transcribing audio",
        detail=window_transcription_stage_detail(
            window_count=len(windows),
            total_duration_sec=media_asset.duration_sec,
            elapsed_sec=transcribe_elapsed_sec,
        ),
    )

    reporter.stage_started("reconcile", "Reconciling transcript windows")
    timer = start_stage_timer(stage_timings, "reconcile")
    transcription, reconciliation_stats = reconcile_decoded_windows(decoded_windows)
    timer.finish()
    asr_pipeline.reconciliation_duplicate_segments_dropped = (
        reconciliation_stats.duplicate_segments_dropped
    )
    asr_pipeline.reconciliation_boundary_fixes = reconciliation_stats.boundary_fixes
    asr_pipeline.system_info = transcriber.system_info
    reporter.stage_finished(
        "reconcile",
        "Reconciling transcript windows",
        detail=count_label(len(transcription.segments), "segment"),
    )

    write_json(layout.transcript_path, transcription.model_dump(mode="json"))
    normalized_transcription = normalize_transcription(transcription)
    return AsrPipelineResult(
        transcription=transcription, normalized_transcription=normalized_transcription
    )
