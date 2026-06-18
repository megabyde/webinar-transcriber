"""Pipeline orchestration for one webinar transcription run."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from webinar_transcriber.asr import ASR_BACKEND_NAME, WhisperCppTranscriber, plan_inference_windows
from webinar_transcriber.diagnostics import write_run_diagnostics
from webinar_transcriber.diarization import DIARIZATION_MODEL, assign_speakers
from webinar_transcriber.export import write_docx_report, write_json_report, write_markdown_report
from webinar_transcriber.export.formatting import format_count, format_rtf, join_detail
from webinar_transcriber.io import write_json
from webinar_transcriber.llm import LlmProcessingError
from webinar_transcriber.media import probe_media
from webinar_transcriber.models import (
    AsrPipelineDiagnostics,
    DiarizationDiagnostics,
    LlmDiagnostics,
    VideoAsset,
    average_duration_sec,
)
from webinar_transcriber.normalized_audio import load_normalized_audio, write_transcription_audio
from webinar_transcriber.paths import create_run_layout
from webinar_transcriber.segmentation import detect_speech_regions, normalized_audio_duration
from webinar_transcriber.structure import build_report
from webinar_transcriber.transcript.normalize import normalize_transcription
from webinar_transcriber.transcript.reconcile import reconcile_decoded_windows
from webinar_transcriber.video import detect_scenes, estimated_scene_sample_count

if TYPE_CHECKING:
    from collections.abc import Iterator

    from webinar_transcriber.diarization import SherpaOnnxDiarizer
    from webinar_transcriber.llm import LlmReportMetadataResult
    from webinar_transcriber.llm.processor import InstructorLLMProcessor
    from webinar_transcriber.models import (
        Diagnostics,
        MediaAsset,
        ReportDocument,
        Scene,
        TranscriptionResult,
    )
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.ui import StageHandle, StageReporter


@dataclass(frozen=True)
class ProcessArtifacts:
    """Runtime artifacts returned from a processing run."""

    layout: RunLayout
    media_asset: MediaAsset
    transcription: TranscriptionResult
    report: ReportDocument
    diagnostics: Diagnostics


@dataclass
class RunContext:
    """Mutable state and recorded diagnostics for one processing run."""

    reporter: StageReporter
    stage_timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    current_stage: str | None = None
    item_counts: dict[str, int] = field(default_factory=dict)
    asr_pipeline: AsrPipelineDiagnostics | None = None
    diarization: DiarizationDiagnostics | None = None
    llm: LlmDiagnostics | None = None

    def record_warning(self, message: str) -> None:
        """Record and report one run warning."""
        self.warnings.append(message)
        self.reporter.warn(message)

    @contextmanager
    def stage(
        self, key: str, label: str, *, total: float | None = None, detail: str | None = None
    ) -> Iterator[StageHandle]:
        """Track timing for one stage and surface it through the reporter."""
        self.current_stage = key
        with self.reporter.track(key, label, total=total, detail=detail) as handle:
            try:
                yield handle
            finally:
                self.stage_timings[key] = handle.elapsed_sec()


def process_input(
    input_path: Path,
    *,
    threads: int,
    transcriber: WhisperCppTranscriber,
    reporter: StageReporter,
    output_dir: Path | None = None,
    keep_audio: bool = False,
    llm_processor: InstructorLLMProcessor | None = None,
    diarizer: SherpaOnnxDiarizer | None = None,
    diarization_speaker_count: int | None = None,
) -> ProcessArtifacts:
    """Process a single audio or video file into report artifacts.

    Returns:
        ProcessArtifacts: The completed processing artifacts.
    """
    ctx = RunContext(reporter=reporter)

    with transcriber as active_transcriber:
        layout = create_run_layout(input_path=input_path, output_dir=output_dir)
        active_transcriber.set_log_path(layout.run_dir / "whisper-cpp.log")
        ctx.reporter.begin_run(input_path)
        try:
            with ctx.stage("probe_media", "Probing media") as st:
                media_asset = probe_media(input_path)
                write_json(layout.metadata_path, asdict(media_asset))
                st.update(detail=f"{media_asset.media_type.value}, {media_asset.duration_sec:.1f}s")

            with tempfile.TemporaryDirectory(prefix="webinar-transcriber-audio-") as temp_dir:
                audio_total = max(media_asset.duration_sec, 1.0)
                with ctx.stage(
                    "prepare_transcription_audio", "Preparing audio", total=audio_total
                ) as st:
                    audio_path = write_transcription_audio(
                        input_path,
                        Path(temp_dir) / f"{input_path.stem}.wav",
                        progress_callback=lambda completed: st.update(
                            completed=min(completed, audio_total)
                        ),
                    )
                    st.update(completed=audio_total, detail=audio_path.name)

                transcription = _run_asr_pipeline(
                    audio_path=audio_path,
                    media_duration_sec=media_asset.duration_sec,
                    transcriber=active_transcriber,
                    layout=layout,
                    ctx=ctx,
                    threads=threads,
                    diarizer=diarizer,
                    diarization_speaker_count=diarization_speaker_count,
                )

                if keep_audio:
                    with ctx.stage(
                        "save_transcription_audio", "Saving transcription audio", total=audio_total
                    ) as st:
                        preserved_audio_path = layout.transcription_audio_path
                        write_transcription_audio(
                            audio_path,
                            preserved_audio_path,
                            audio_format="mp3",
                            progress_callback=lambda completed: st.update(
                                completed=min(completed, audio_total)
                            ),
                        )
                        st.update(completed=audio_total, detail=preserved_audio_path.name)

            report = _run_report_phase(
                input_path=input_path,
                layout=layout,
                media_asset=media_asset,
                transcription=transcription,
                llm_processor=llm_processor,
                ctx=ctx,
            )

            diagnostics = write_run_diagnostics(layout, ctx, status="succeeded")
            artifacts = ProcessArtifacts(
                layout=layout,
                media_asset=media_asset,
                transcription=transcription,
                report=report,
                diagnostics=diagnostics,
            )
            ctx.reporter.complete_run(artifacts)
            return artifacts
        except Exception as ex:
            # Diagnostics are best effort on failure; never mask the original error.
            with suppress(Exception):
                write_run_diagnostics(
                    layout, ctx, status="failed", failed_stage=ctx.current_stage, error=str(ex)
                )
            raise


def _run_asr_pipeline(
    *,
    audio_path: Path,
    media_duration_sec: float,
    transcriber: WhisperCppTranscriber,
    layout: RunLayout,
    ctx: RunContext,
    threads: int,
    diarizer: SherpaOnnxDiarizer | None,
    diarization_speaker_count: int | None,
) -> TranscriptionResult:
    """Run ASR and optional diarization, recording diagnostics on the context.

    Returns:
        TranscriptionResult: The reconciled transcription.
    """
    with ctx.stage("prepare_asr", "Preparing ASR model", detail=transcriber.model_name) as st:
        transcriber.prepare_model()
        st.update(detail=str(transcriber))

    audio_samples = load_normalized_audio(audio_path)
    audio_duration_sec = normalized_audio_duration(audio_samples)

    with ctx.stage(
        "vad", "Detecting speech regions", total=audio_duration_sec, detail="0 regions"
    ) as st:

        def on_vad_progress(completed: float, count: int) -> None:
            st.update(completed=completed, detail=format_count(count, "region"))

        speech_regions, vad_warnings = detect_speech_regions(
            audio_samples, threads=threads, progress_callback=on_vad_progress
        )
        st.update(
            completed=audio_duration_sec,
            detail=join_detail(
                format_count(len(speech_regions), "region"),
                format_rtf(audio_duration_sec, st.elapsed_sec()),
            ),
        )
    for warning in vad_warnings:
        ctx.record_warning(warning)
    write_json(layout.speech_regions_path, [asdict(region) for region in speech_regions])
    ctx.item_counts["vad_regions"] = len(speech_regions)

    windows = plan_inference_windows(speech_regions)
    ctx.item_counts["windows"] = len(windows)

    with ctx.stage(
        "transcribe", "Transcribing audio", total=len(windows), detail="0 segments"
    ) as st:

        def on_transcribe_progress(completed: int, count: int) -> None:
            st.update(completed=completed, detail=format_count(count, "segment"))

        decoded_windows = transcriber.transcribe_inference_windows(
            audio_samples,
            windows,
            progress_callback=on_transcribe_progress,
            warning_callback=ctx.record_warning,
        )
        write_json(layout.decoded_windows_path, [window.to_json() for window in decoded_windows])
        segment_count = sum(len(window.segments) for window in decoded_windows)
        st.update(
            completed=len(windows),
            detail=join_detail(
                format_count(segment_count, "segment"),
                format_rtf(media_duration_sec, st.elapsed_sec()),
            ),
        )
    transcription = reconcile_decoded_windows(decoded_windows)
    ctx.item_counts["transcript_segments"] = len(transcription.segments)

    if diarizer is not None:
        # Model prep and the segmentation pass run as opaque native calls that freeze the display,
        # so the stage opens already labeled "analyzing audio" rather than briefly showing
        # "preparing model" — which would otherwise be the frame left stuck on screen.
        with ctx.stage(
            "diarize", "Diarizing speakers", total=100.0, detail="analyzing audio"
        ) as st:
            diarizer.prepare(speaker_count=diarization_speaker_count)

            def on_diarize_progress(processed: int, total: int) -> None:
                embedding_progress = 60.0 * min(processed, total) / total
                st.update(completed=35.0 + embedding_progress, detail="embedding speakers")

            speaker_turns = diarizer.diarize(audio_path, progress_callback=on_diarize_progress)
            write_json(layout.diarization_path, [asdict(turn) for turn in speaker_turns])
            speaker_count = len({turn.speaker for turn in speaker_turns})
            transcription = replace(
                transcription, segments=assign_speakers(transcription.segments, speaker_turns)
            )
            st.update(
                completed=100.0,
                detail=join_detail(
                    format_count(speaker_count, "speaker"),
                    format_count(len(speaker_turns), "turn"),
                    format_rtf(audio_duration_sec, st.elapsed_sec()),
                ),
            )
        ctx.diarization = DiarizationDiagnostics(
            speaker_count=speaker_count,
            turn_count=len(speaker_turns),
            model=DIARIZATION_MODEL,
            system_info=diarizer.system_info,
        )

    write_json(layout.transcript_path, transcription.to_json())

    ctx.asr_pipeline = AsrPipelineDiagnostics(
        backend=ASR_BACKEND_NAME,
        model=transcriber.model_name,
        threads=threads,
        average_window_duration_sec=average_duration_sec(windows),
        normalized_audio_duration_sec=audio_duration_sec,
        system_info=transcriber.system_info,
    )
    return transcription


def _run_report_phase(
    *,
    input_path: Path,
    layout: RunLayout,
    media_asset: MediaAsset,
    transcription: TranscriptionResult,
    llm_processor: InstructorLLMProcessor | None,
    ctx: RunContext,
) -> ReportDocument:
    """Build, polish, and export the report, recording diagnostics on the context.

    Returns:
        ReportDocument: The final exported report.
    """
    normalized_transcription = normalize_transcription(transcription)
    ctx.item_counts["normalized_transcript_segments"] = len(normalized_transcription.segments)

    scenes: list[Scene] = []

    if isinstance(media_asset, VideoAsset):
        scenes = _detect_video_scenes(
            input_path=input_path, layout=layout, media_asset=media_asset, ctx=ctx
        )

    ctx.item_counts["scenes"] = len(scenes)
    ctx.item_counts["frames"] = sum(scene.image_path is not None for scene in scenes)
    report = build_report(media_asset, normalized_transcription, scenes=scenes)

    if llm_processor is not None:
        report = _polish_report(report, llm_processor=llm_processor, ctx=ctx)
    ctx.item_counts["report_sections"] = len(report.sections)

    _export_report(report, layout=layout, ctx=ctx)
    return report


def _detect_video_scenes(
    *,
    input_path: Path,
    layout: RunLayout,
    media_asset: VideoAsset,
    ctx: RunContext,
) -> list[Scene]:
    """Detect scenes and save each scene's representative frame in one decode pass.

    Returns:
        list[Scene]: Detected scenes, each carrying its representative frame image path.
    """
    scene_sample_total = estimated_scene_sample_count(media_asset.duration_sec)
    with ctx.stage(
        "detect_scenes", "Detecting scenes", total=scene_sample_total, detail="0 scenes"
    ) as st:

        def on_scene_progress(completed: float, count: int) -> None:
            st.update(completed=completed, detail=format_count(count, "scene"))

        scenes = detect_scenes(
            input_path,
            layout.frames_dir,
            media_asset.duration_sec,
            progress_callback=on_scene_progress,
        )
        st.update(completed=scene_sample_total, detail=format_count(len(scenes), "scene"))

    write_json(layout.scenes_path, [asdict(scene) for scene in scenes])
    return scenes


def _export_report(report: ReportDocument, *, layout: RunLayout, ctx: RunContext) -> None:
    """Write the Markdown, DOCX, and JSON artifacts for the report."""
    with ctx.stage("export", "Writing artifacts") as st:
        write_markdown_report(report, layout.markdown_report_path)
        write_docx_report(report, layout.docx_report_path)
        write_json_report(report, layout.json_report_path)
        st.update(detail="report.md | report.docx | report.json")


def _polish_report(
    report: ReportDocument,
    *,
    llm_processor: InstructorLLMProcessor,
    ctx: RunContext,
) -> ReportDocument:
    provider_name = llm_processor.provider_name
    model_name = llm_processor.model_name
    section_count = len(report.sections)
    worker_count = llm_processor.polish_worker_count(section_count)
    runtime_detail = join_detail(provider_name, model_name)

    with ctx.stage(
        "llm_report_sections",
        "Polishing sections with LLM",
        total=section_count,
        detail=join_detail(runtime_detail, format_count(worker_count, "worker")),
    ) as st:

        def on_section_progress(advance: int) -> None:
            st.update(advance=advance)

        try:
            section_result = llm_processor.polish_report_sections(
                report, progress_callback=on_section_progress
            )
        except LlmProcessingError as ex:
            return _record_llm_fallback(
                report, ex, ctx=ctx, st=st, runtime_detail=runtime_detail, model_name=model_name
            )
        section_elapsed_sec = st.elapsed_sec()
        st.update(detail=format_count(section_count, "section"))
    for warning in section_result.warnings:
        ctx.record_warning(warning)

    with ctx.stage(
        "llm_report_metadata", "Polishing report summary with LLM", detail=runtime_detail
    ) as st:
        try:
            metadata_result = llm_processor.polish_report_metadata(
                report, section_transcripts=section_result.section_transcripts
            )
        except LlmProcessingError as ex:
            return _record_llm_fallback(
                report,
                ex,
                ctx=ctx,
                st=st,
                runtime_detail=runtime_detail,
                model_name=model_name,
                prior_elapsed_sec=section_elapsed_sec,
                response_metadata=section_result.response_metadata,
            )
        metadata_elapsed_sec = st.elapsed_sec()
        st.update(detail=_metadata_detail(metadata_result))

    polished_report = replace(
        report,
        summary=metadata_result.summary,
        action_items=metadata_result.action_items,
        sections=[
            replace(
                section,
                title=metadata_result.section_titles.get(section.id, section.title),
                tldr=section_result.section_tldrs.get(section.id, section.tldr),
                transcript_text=section_result.section_transcripts.get(
                    section.id, section.transcript_text
                ),
            )
            for section in report.sections
        ],
    )
    ctx.llm = LlmDiagnostics(
        model=model_name,
        report_status="applied",
        report_latency_sec=section_elapsed_sec + metadata_elapsed_sec,
        response_metadata=[
            *section_result.response_metadata,
            *metadata_result.response_metadata,
        ],
    )
    return polished_report


def _record_llm_fallback(
    report: ReportDocument,
    ex: LlmProcessingError,
    *,
    ctx: RunContext,
    st: StageHandle,
    runtime_detail: str,
    model_name: str,
    prior_elapsed_sec: float = 0.0,
    response_metadata: list[dict[str, object]] | None = None,
) -> ReportDocument:
    """Record an LLM-stage fallback (warning + diagnostics) and return the unpolished report.

    Returns:
        ReportDocument: The original, unpolished report.
    """
    ctx.record_warning(str(ex))
    st.update(detail=join_detail(runtime_detail, "fallback"))
    ctx.llm = LlmDiagnostics(
        model=model_name,
        report_status="fallback",
        report_latency_sec=prior_elapsed_sec + st.elapsed_sec(),
        response_metadata=response_metadata or [],
    )
    return report


def _metadata_detail(metadata_result: LlmReportMetadataResult) -> str:
    parts = []
    if metadata_result.summary:
        parts.append(format_count(len(metadata_result.summary), "summary bullet"))
    if metadata_result.action_items:
        parts.append(format_count(len(metadata_result.action_items), "action item"))
    if metadata_result.section_titles:
        parts.append(f"{format_count(len(metadata_result.section_titles), 'title')} updated")
    return join_detail(*parts)


__all__ = [
    "ProcessArtifacts",
    "RunContext",
    "process_input",
]
