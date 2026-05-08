"""ASR pipeline helpers used by the top-level processor orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from webinar_transcriber.models import (
    AsrPipelineDiagnostics,
    InferenceWindow,
    SpeechRegion,
    TranscriptionResult,
)
from webinar_transcriber.normalized_audio import load_normalized_audio
from webinar_transcriber.segmentation import detect_speech_regions, normalized_audio_duration
from webinar_transcriber.transcript import reconcile_decoded_windows
from webinar_transcriber.transcript.normalize import normalize_transcription

from .support import (
    asr_runtime_detail,
    count_label,
    progress_stage,
    stage,
    window_transcription_stage_detail,
    write_json,
)

if TYPE_CHECKING:
    from pathlib import Path

    from webinar_transcriber.asr import WhisperCppTranscriber
    from webinar_transcriber.models import MediaAsset
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.segmentation import VadSettings

    from .support import ProgressStageHandle
    from .types import RunContext

MAX_INFERENCE_WINDOW_SEC = 28.0
INFERENCE_WINDOW_OVERLAP_SEC = 2.0


@dataclass(frozen=True)
class AsrPipelineResult:
    """Raw and normalized transcription outputs from the ASR pipeline."""

    transcription: TranscriptionResult
    normalized_transcription: TranscriptionResult
    asr_pipeline: AsrPipelineDiagnostics


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
    language: str | None,
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
            settings=vad,
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

    windows = plan_inference_windows(speech_regions)
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
            completed_sec: float, segment_count: int, handle: ProgressStageHandle = st
        ) -> None:
            handle.advance_to(completed_sec, detail=count_label(segment_count, "segment"))

        decoded_windows = transcriber.transcribe_inference_windows(
            audio_samples, windows, language=language, progress_callback=on_window_completed
        )
        write_json(
            layout.decoded_windows_path,
            {"decoded_windows": [asdict(window) for window in decoded_windows]},
        )
        st.advance_to(media_asset.duration_sec)
        st.detail = window_transcription_stage_detail(
            window_count=len(windows),
            total_duration_sec=media_asset.duration_sec,
            elapsed_sec=st.elapsed_sec(),
        )

    transcription = reconcile_decoded_windows(decoded_windows)

    with stage(ctx, "normalize_transcript", "Normalizing transcript") as st:
        write_json(layout.transcript_path, asdict(transcription))
        normalized_transcription = normalize_transcription(transcription)
        st.detail = count_label(len(normalized_transcription.segments), "segment")

    asr_pipeline = AsrPipelineDiagnostics(
        vad_enabled=vad.enabled,
        threads=transcriber.threads,
        normalized_audio_duration_sec=normalized_audio_duration_sec,
        vad_region_count=vad_region_count,
        carryover_enabled=carryover_enabled,
        window_count=window_count,
        average_window_duration_sec=average_window_duration_sec,
        system_info=transcriber.system_info,
    )
    return AsrPipelineResult(
        transcription=transcription,
        normalized_transcription=normalized_transcription,
        asr_pipeline=asr_pipeline,
    )


def plan_inference_windows(
    speech_regions: list[SpeechRegion],
    *,
    max_window_sec: float = MAX_INFERENCE_WINDOW_SEC,
    overlap_sec: float = INFERENCE_WINDOW_OVERLAP_SEC,
) -> list[InferenceWindow]:
    """Plan bounded Whisper inference windows from speech regions.

    Returns:
        list[InferenceWindow]: Ordered windows with overlap inside long regions.
    """
    windows: list[InferenceWindow] = []
    safe_max_window_sec = max(max_window_sec, 0.1)
    safe_overlap_sec = min(max(overlap_sec, 0.0), safe_max_window_sec / 2.0)

    for region_index, region in enumerate(speech_regions):
        if region.end_sec <= region.start_sec:
            continue

        window_start_sec = region.start_sec
        while window_start_sec < region.end_sec:
            window_end_sec = min(region.end_sec, window_start_sec + safe_max_window_sec)
            windows.append(
                InferenceWindow(
                    window_id=f"window-{len(windows) + 1}",
                    region_index=region_index,
                    start_sec=window_start_sec,
                    end_sec=window_end_sec,
                )
            )
            if window_end_sec >= region.end_sec:
                break
            window_start_sec = max(window_start_sec, window_end_sec - safe_overlap_sec)

    return windows
