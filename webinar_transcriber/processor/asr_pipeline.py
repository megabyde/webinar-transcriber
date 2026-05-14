"""ASR pipeline helpers used by the top-level processor orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from webinar_transcriber.asr import ASR_BACKEND_NAME
from webinar_transcriber.diarization import DIARIZATION_MODEL, SherpaOnnxDiarizer, assign_speakers
from webinar_transcriber.models import (
    AsrPipelineDiagnostics,
    DiarizationDiagnostics,
    InferenceWindow,
    SpeakerTurn,
    SpeechRegion,
    TimelineSpan,
    TranscriptionResult,
)
from webinar_transcriber.normalized_audio import load_normalized_audio
from webinar_transcriber.segmentation import detect_speech_regions, normalized_audio_duration
from webinar_transcriber.transcript import reconcile_decoded_windows
from webinar_transcriber.transcript.normalize import normalize_transcription

from .support import (
    asr_runtime_detail,
    count_label,
    counting_progress,
    progress_stage,
    realtime_factor_detail,
    stage,
    transcription_stage_detail,
    write_json,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np

    from webinar_transcriber.asr import WhisperCppTranscriber
    from webinar_transcriber.diarization import Diarizer
    from webinar_transcriber.models import MediaAsset
    from webinar_transcriber.paths import RunLayout

    from .support import ProgressStageHandle
    from .types import RunContext

# sherpa-onnx reports progress only after its full-audio segmentation pass.
DIARIZATION_SEGMENTATION_PROGRESS_PERCENT = 35.0
DIARIZATION_EMBEDDING_PROGRESS_PERCENT = 60.0
DIARIZATION_INITIAL_PROGRESS_PERCENT = 1.0
INFERENCE_WINDOW_DURATION_SEC = 28.0
INFERENCE_WINDOW_OVERLAP_SEC = 2.0


@dataclass(frozen=True)
class AsrPipelineResult:
    """Raw and normalized transcription outputs from the ASR pipeline."""

    transcription: TranscriptionResult
    normalized_transcription: TranscriptionResult
    asr_pipeline: AsrPipelineDiagnostics
    diarization: DiarizationDiagnostics | None = None


def run_asr_pipeline(
    *,
    audio_path: Path,
    media_asset: MediaAsset,
    transcriber: WhisperCppTranscriber,
    layout: RunLayout,
    ctx: RunContext,
    vad: bool,
    carryover_enabled: bool,
    language: str | None,
    diarize: bool,
    diarize_speakers: int | None,
    diarizer: Diarizer | None = None,
) -> AsrPipelineResult:
    """Run the deterministic local ASR pipeline and persist intermediate artifacts.

    Returns:
        AsrPipelineResult: The raw and normalized transcription outputs.
    """
    with stage(ctx, "prepare_asr", "Preparing ASR model") as st:
        transcriber.prepare_model()
        st.set_detail(asr_runtime_detail(transcriber))

    audio_samples, _ = load_normalized_audio(audio_path)
    speech_regions = _detect_speech_regions_stage(
        audio_samples=audio_samples, threads=transcriber.threads, ctx=ctx, vad=vad, layout=layout
    )
    windows = _plan_inference_windows_stage(ctx=ctx, speech_regions=speech_regions)
    vad_region_count = len(speech_regions)

    window_count = len(windows)
    average_window_duration_sec = average_duration_sec(windows)
    with progress_stage(
        ctx,
        "transcribe",
        "Transcribing audio",
        total=media_asset.duration_sec,
        count_label="s",
        detail="0 segments",
    ) as st:
        decoded_windows = transcriber.transcribe_inference_windows(
            audio_samples,
            windows,
            language=language,
            progress_callback=counting_progress(st, "segment"),
            warning_callback=ctx.record_warning,
        )
        write_json(
            layout.decoded_windows_path,
            [window.to_json() for window in decoded_windows],
        )
        st.advance_to(media_asset.duration_sec)
        st.set_detail(
            transcription_stage_detail(
                segment_count=sum(len(window.segments) for window in decoded_windows),
                total_duration_sec=media_asset.duration_sec,
                elapsed_sec=st.elapsed_sec(),
            )
        )

    transcription = reconcile_decoded_windows(decoded_windows)

    diarization: DiarizationDiagnostics | None = None
    if diarize:
        active_diarizer = diarizer or SherpaOnnxDiarizer(threads=transcriber.threads)
        with stage(ctx, "prepare_diarization", "Preparing diarization model") as st:
            active_diarizer.prepare(speaker_count=diarize_speakers)
            st.set_detail(DIARIZATION_MODEL)
        speaker_turns = _diarize_speakers_stage(
            audio_samples=audio_samples, ctx=ctx, layout=layout, diarizer=active_diarizer
        )
        transcription = _assign_speakers_stage(
            ctx=ctx, transcription=transcription, speaker_turns=speaker_turns
        )
        diarization = DiarizationDiagnostics(
            speaker_count=len({turn.speaker for turn in speaker_turns}),
            turn_count=len(speaker_turns),
            average_turn_duration_sec=average_duration_sec(speaker_turns),
            model=DIARIZATION_MODEL,
            system_info=active_diarizer.system_info,
        )

    with stage(ctx, "normalize_transcript", "Normalizing transcript") as st:
        normalized_transcription = normalize_transcription(transcription)
        st.set_detail(count_label(len(normalized_transcription.segments), "segment"))

    write_json(layout.transcript_path, transcription.to_json())

    asr_pipeline = AsrPipelineDiagnostics(
        backend=ASR_BACKEND_NAME,
        model=transcriber.model_name,
        vad_enabled=vad,
        threads=transcriber.threads,
        vad_region_count=vad_region_count,
        carryover_enabled=carryover_enabled,
        window_count=window_count,
        average_window_duration_sec=average_window_duration_sec,
        normalized_audio_duration_sec=normalized_audio_duration(audio_samples),
        system_info=transcriber.system_info,
    )
    return AsrPipelineResult(
        transcription=transcription,
        normalized_transcription=normalized_transcription,
        asr_pipeline=asr_pipeline,
        diarization=diarization,
    )


def _detect_speech_regions_stage(
    *,
    audio_samples: np.ndarray,
    threads: int,
    ctx: RunContext,
    vad: bool,
    layout: RunLayout,
) -> list[SpeechRegion]:
    audio_duration_sec = normalized_audio_duration(audio_samples)
    with progress_stage(
        ctx,
        "vad",
        "Detecting speech regions",
        total=audio_duration_sec,
        count_label="s",
        detail="0 regions",
    ) as st:
        speech_regions, vad_warnings = detect_speech_regions(
            audio_samples,
            enabled=vad,
            threads=threads,
            progress_callback=counting_progress(st, "region"),
        )
        vad_detail_parts = [count_label(len(speech_regions), "region")]
        if rtf := realtime_factor_detail(
            total_duration_sec=audio_duration_sec, elapsed_sec=st.elapsed_sec()
        ):
            vad_detail_parts.append(rtf)
        vad_detail = " | ".join(vad_detail_parts)
        st.advance_to(audio_duration_sec, detail=vad_detail)
        st.set_detail(vad_detail)

    for warning in vad_warnings:
        ctx.record_warning(warning)
    write_json(
        layout.speech_regions_path,
        [asdict(region) for region in speech_regions],
    )
    return speech_regions


def _plan_inference_windows_stage(
    *, ctx: RunContext, speech_regions: list[SpeechRegion]
) -> list[InferenceWindow]:
    with stage(ctx, "plan_windows", "Planning ASR windows") as st:
        windows = plan_inference_windows(speech_regions)
        st.set_detail(count_label(len(windows), "window"))
    return windows


def _diarize_speakers_stage(
    *,
    audio_samples: np.ndarray,
    ctx: RunContext,
    layout: RunLayout,
    diarizer: Diarizer,
) -> list[SpeakerTurn]:
    with progress_stage(
        ctx,
        "diarize",
        "Diarizing speakers",
        total=100.0,
        count_label="%",
        detail="analyzing audio",
    ) as st:
        st.advance_to(DIARIZATION_INITIAL_PROGRESS_PERCENT, detail="analyzing audio")

        def on_diarization_progress(
            processed_chunks: int,
            total_chunks: int,
            handle: ProgressStageHandle = st,
        ) -> None:
            if total_chunks <= 0:
                return
            embedding_progress = (
                DIARIZATION_EMBEDDING_PROGRESS_PERCENT
                * min(processed_chunks, total_chunks)
                / total_chunks
            )
            handle.advance_to(
                DIARIZATION_SEGMENTATION_PROGRESS_PERCENT + embedding_progress,
                detail="embedding speakers",
            )

        speaker_turns = diarizer.diarize(audio_samples, progress_callback=on_diarization_progress)
        write_json(layout.diarization_path, [asdict(turn) for turn in speaker_turns])
        speaker_count = len({turn.speaker for turn in speaker_turns})
        turn_count = len(speaker_turns)
        detail = " | ".join([
            count_label(speaker_count, "speaker"),
            count_label(turn_count, "turn"),
        ])
        if rtf := realtime_factor_detail(
            total_duration_sec=normalized_audio_duration(audio_samples),
            elapsed_sec=st.elapsed_sec(),
        ):
            detail = f"{detail} | {rtf}"
        st.advance_to(100.0, detail=detail)
        st.set_detail(detail)
    return speaker_turns


def _assign_speakers_stage(
    *,
    ctx: RunContext,
    transcription: TranscriptionResult,
    speaker_turns: list[SpeakerTurn],
) -> TranscriptionResult:
    with stage(ctx, "assign_speakers", "Assigning speakers") as st:
        segments = assign_speakers(transcription.segments, speaker_turns)
        assigned_count = sum(segment.speaker is not None for segment in segments)
        st.set_detail(count_label(assigned_count, "segment"))
    return TranscriptionResult(detected_language=transcription.detected_language, segments=segments)


def average_duration_sec(items: Sequence[TimelineSpan]) -> float | None:
    """Return the average duration for timeline-bounded items."""
    return sum(item.duration_sec for item in items) / len(items) if items else None


def plan_inference_windows(
    speech_regions: list[SpeechRegion],
) -> list[InferenceWindow]:
    """Plan bounded Whisper inference windows from speech regions.

    Returns:
        list[InferenceWindow]: Ordered windows with overlap inside long regions.
    """
    windows: list[InferenceWindow] = []

    for region_index, region in enumerate(speech_regions):
        if region.duration_sec <= 0:
            continue

        window_start_sec = region.start_sec
        while window_start_sec < region.end_sec:
            window_end_sec = min(region.end_sec, window_start_sec + INFERENCE_WINDOW_DURATION_SEC)
            windows.append(
                InferenceWindow(
                    id=f"window-{len(windows) + 1}",
                    region_index=region_index,
                    start_sec=window_start_sec,
                    end_sec=window_end_sec,
                )
            )
            if window_end_sec >= region.end_sec:
                break
            window_start_sec = max(window_start_sec, window_end_sec - INFERENCE_WINDOW_OVERLAP_SEC)

    return windows
