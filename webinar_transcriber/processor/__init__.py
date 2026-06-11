"""Pipeline orchestration for one webinar transcription run."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from webinar_transcriber.asr import ASR_BACKEND_NAME, WhisperCppTranscriber
from webinar_transcriber.diagnostics import write_run_diagnostics
from webinar_transcriber.diarization import DIARIZATION_MODEL, assign_speakers
from webinar_transcriber.export import write_docx_report, write_json_report, write_markdown_report
from webinar_transcriber.io import write_json
from webinar_transcriber.llm import LlmProcessingError
from webinar_transcriber.media import probe_media
from webinar_transcriber.models import (
    AsrPipelineDiagnostics,
    DiarizationDiagnostics,
    LlmDiagnostics,
    TranscriptionResult,
    VideoAsset,
)
from webinar_transcriber.normalized_audio import (
    load_normalized_audio,
    prepared_transcription_audio,
    preserve_transcription_audio,
)
from webinar_transcriber.paths import create_run_layout
from webinar_transcriber.segmentation import detect_speech_regions, normalized_audio_duration
from webinar_transcriber.structure import align_by_time, build_report
from webinar_transcriber.transcript.normalize import normalize_transcription
from webinar_transcriber.transcript.reconcile import reconcile_decoded_windows
from webinar_transcriber.ui import StageReporter
from webinar_transcriber.video import (
    detect_scenes,
    estimated_scene_sample_count,
    extract_representative_frames,
)

from .stages import (
    INFERENCE_WINDOW_DURATION_SEC,
    INFERENCE_WINDOW_OVERLAP_SEC,
    average_duration_sec,
    plan_inference_windows,
)

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
        SceneFrame,
    )
    from webinar_transcriber.paths import RunLayout
    from webinar_transcriber.ui import StageHandle


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
    output_dir: Path | None = None,
    asr_model: str | None = None,
    language: str | None = None,
    keep_audio: bool = False,
    llm_processor: InstructorLLMProcessor | None = None,
    diarizer: SherpaOnnxDiarizer | None = None,
    diarization_speaker_count: int | None = None,
    transcriber: WhisperCppTranscriber | None = None,
    reporter: StageReporter | None = None,
) -> ProcessArtifacts:
    """Process a single audio or video file into report artifacts.

    Returns:
        ProcessArtifacts: The completed processing artifacts.
    """
    ctx = RunContext(reporter=reporter or _silent_reporter())
    transcriber = transcriber or WhisperCppTranscriber(
        model_name=asr_model, threads=threads, language=language
    )

    with transcriber as active_transcriber:
        layout = create_run_layout(input_path=input_path, output_dir=output_dir)
        active_transcriber.set_log_path(layout.run_dir / "whisper-cpp.log")
        ctx.reporter.begin_run(input_path)
        try:
            with ctx.stage("probe_media", "Probing media") as st:
                media_asset = probe_media(input_path)
                write_json(layout.metadata_path, asdict(media_asset))
                st.update(detail=f"{media_asset.media_type.value}, {media_asset.duration_sec:.1f}s")

            with ExitStack() as audio_scope:
                audio_total = max(media_asset.duration_sec, 1.0)
                with ctx.stage(
                    "prepare_transcription_audio", "Preparing audio", total=audio_total
                ) as st:
                    audio_path = audio_scope.enter_context(
                        prepared_transcription_audio(
                            input_path,
                            progress_callback=lambda completed: st.update(
                                completed=min(completed, audio_total)
                            ),
                        )
                    )
                    st.update(completed=audio_total, detail=audio_path.name)

                transcription = _run_asr_pipeline(
                    audio_path=audio_path,
                    media_asset=media_asset,
                    transcriber=active_transcriber,
                    layout=layout,
                    ctx=ctx,
                    threads=threads,
                    language=language,
                    diarizer=diarizer,
                    diarization_speaker_count=diarization_speaker_count,
                )

                if keep_audio:
                    with ctx.stage("save_transcription_audio", "Saving transcription audio") as st:
                        preserved_audio_path = layout.transcription_audio_path()
                        preserve_transcription_audio(audio_path, preserved_audio_path)
                        st.update(detail=preserved_audio_path.name)

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
            write_run_diagnostics(
                layout,
                ctx,
                status="failed",
                failed_stage=ctx.current_stage,
                error=str(ex),
                suppress_errors=True,
            )
            raise


def _run_asr_pipeline(
    *,
    audio_path: Path,
    media_asset: MediaAsset,
    transcriber: WhisperCppTranscriber,
    layout: RunLayout,
    ctx: RunContext,
    threads: int,
    language: str | None,
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

    audio_samples, _ = load_normalized_audio(audio_path)
    audio_duration_sec = normalized_audio_duration(audio_samples)

    with ctx.stage(
        "vad", "Detecting speech regions", total=audio_duration_sec, detail="0 regions"
    ) as st:

        def on_vad_progress(completed: float, count: int) -> None:
            st.update(completed=completed, detail=_count(count, "region"))

        speech_regions, vad_warnings = detect_speech_regions(
            audio_samples, threads=threads, progress_callback=on_vad_progress
        )
        st.update(
            completed=audio_duration_sec,
            detail=_join_detail(
                _count(len(speech_regions), "region"), _rtf(audio_duration_sec, st.elapsed_sec())
            ),
        )
    for warning in vad_warnings:
        ctx.record_warning(warning)
    write_json(layout.speech_regions_path, [asdict(region) for region in speech_regions])
    ctx.item_counts["vad_regions"] = len(speech_regions)

    windows = plan_inference_windows(speech_regions)
    ctx.item_counts["windows"] = len(windows)

    with ctx.stage(
        "transcribe", "Transcribing audio", total=float(len(windows)), detail="0 segments"
    ) as st:

        def on_transcribe_progress(completed: int, count: int) -> None:
            st.update(completed=float(completed), detail=_count(count, "segment"))

        decoded_windows = transcriber.transcribe_inference_windows(
            audio_samples,
            windows,
            language=language,
            progress_callback=on_transcribe_progress,
            warning_callback=ctx.record_warning,
        )
        write_json(layout.decoded_windows_path, [window.to_json() for window in decoded_windows])
        segment_count = sum(len(window.segments) for window in decoded_windows)
        st.update(
            completed=float(len(windows)),
            detail=_join_detail(
                _count(segment_count, "segment"), _rtf(media_asset.duration_sec, st.elapsed_sec())
            ),
        )
    transcription = reconcile_decoded_windows(decoded_windows)
    ctx.item_counts["transcript_segments"] = len(transcription.segments)

    if diarizer is not None:
        with ctx.stage(
            "diarize", "Diarizing speakers", total=100.0, detail="preparing model"
        ) as st:
            diarizer.prepare(speaker_count=diarization_speaker_count)
            st.update(completed=1.0, detail="analyzing audio")

            def on_diarize_progress(processed: int, total: int) -> None:
                if total <= 0:
                    return
                embedding_progress = 60.0 * min(processed, total) / total
                st.update(completed=35.0 + embedding_progress, detail="embedding speakers")

            speaker_turns = diarizer.diarize(audio_samples, progress_callback=on_diarize_progress)
            write_json(layout.diarization_path, [asdict(turn) for turn in speaker_turns])
            speaker_count = len({turn.speaker for turn in speaker_turns})
            transcription = _assign_speakers(transcription, speaker_turns)
            st.update(
                completed=100.0,
                detail=_join_detail(
                    _count(speaker_count, "speaker"),
                    _count(len(speaker_turns), "turn"),
                    _rtf(audio_duration_sec, st.elapsed_sec()),
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
        vad_region_count=len(speech_regions),
        window_count=len(windows),
        average_window_duration_sec=average_duration_sec(windows),
        normalized_audio_duration_sec=audio_duration_sec,
        system_info=transcriber.system_info,
    )
    return transcription


def _assign_speakers(
    transcription: TranscriptionResult, speaker_turns: list
) -> TranscriptionResult:
    segments = assign_speakers(transcription.segments, speaker_turns)
    return TranscriptionResult(detected_language=transcription.detected_language, segments=segments)


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
    scene_frames: list[SceneFrame] = []
    alignment_blocks = None

    if isinstance(media_asset, VideoAsset):
        scene_sample_total = float(estimated_scene_sample_count(media_asset.duration_sec))
        with ctx.stage(
            "detect_scenes", "Detecting scenes", total=scene_sample_total, detail="0 scenes"
        ) as st:

            def on_scene_progress(completed: float, count: int) -> None:
                st.update(completed=float(completed), detail=_count(count, "scene"))

            scenes = detect_scenes(
                input_path,
                duration_sec=media_asset.duration_sec,
                progress_callback=on_scene_progress,
            )
            st.update(completed=scene_sample_total, detail=_count(len(scenes), "scene"))
        write_json(layout.scenes_path, [asdict(scene) for scene in scenes])

        with ctx.stage("extract_frames", "Extracting scene frames", total=float(len(scenes))) as st:
            scene_frames = extract_representative_frames(
                input_path,
                scenes,
                layout.frames_dir,
                progress_callback=lambda completed: st.update(completed=float(completed)),
                warning_callback=ctx.record_warning,
            )

        alignment_blocks = align_by_time(normalized_transcription.segments, scenes, scene_frames)

    ctx.item_counts["scenes"] = len(scenes)
    ctx.item_counts["frames"] = len(scene_frames)
    report = build_report(
        media_asset,
        normalized_transcription,
        alignment_blocks=alignment_blocks,
        warnings=ctx.warnings,
    )

    if scene_frames:
        report = _attach_section_images(report, scene_frames, layout.run_dir)

    if llm_processor is not None:
        report = _polish_report(report, llm_processor=llm_processor, ctx=ctx)
    report = replace(report, warnings=list(ctx.warnings))
    ctx.item_counts["report_sections"] = len(report.sections)

    with ctx.stage("export", "Writing artifacts") as st:
        write_markdown_report(report, layout.markdown_report_path)
        write_docx_report(report, layout.docx_report_path, warning_callback=ctx.record_warning)
        report = replace(report, warnings=list(ctx.warnings))
        write_json_report(report, layout.json_report_path)
        st.update(detail="report.md | report.docx | report.json")

    return report


def _attach_section_images(
    report: ReportDocument, scene_frames: list[SceneFrame], run_dir: Path
) -> ReportDocument:
    frame_by_id = {frame.id: frame for frame in scene_frames}
    return replace(
        report,
        sections=[
            replace(section, image_path=_section_image_path(frame, run_dir))
            if section.frame_id and (frame := frame_by_id.get(section.frame_id))
            else section
            for section in report.sections
        ],
    )


def _section_image_path(frame: SceneFrame, run_dir: Path) -> str:
    image_path = Path(frame.image_path)
    try:
        return image_path.relative_to(run_dir).as_posix()
    except ValueError:
        try:
            return image_path.resolve().relative_to(run_dir.resolve()).as_posix()
        except ValueError:
            return frame.image_path


def _polish_report(
    report: ReportDocument,
    *,
    llm_processor: InstructorLLMProcessor,
    ctx: RunContext,
) -> ReportDocument:
    provider_name = llm_processor.provider_name
    model_name = llm_processor.model_name
    polish_plan = llm_processor.report_polish_plan(report)
    runtime_detail = _join_detail(provider_name, model_name)

    with ctx.stage(
        "llm_report_sections",
        "Polishing sections with LLM",
        total=max(float(polish_plan.section_count), 1.0),
        detail=_join_detail(runtime_detail, _count(polish_plan.worker_count, "worker")),
    ) as st:

        def on_section_progress(advance: int) -> None:
            st.update(advance=float(advance))

        try:
            section_result = llm_processor.polish_report_sections_with_progress(
                report, progress_callback=on_section_progress
            )
        except LlmProcessingError as error:
            ctx.record_warning(str(error))
            st.update(detail=_join_detail(runtime_detail, "fallback"))
            ctx.llm = LlmDiagnostics(
                model=model_name,
                report_status="fallback",
                report_latency_sec=st.elapsed_sec(),
            )
            return report
        section_elapsed_sec = st.elapsed_sec()
        st.update(detail=_count(polish_plan.section_count, "section"))
    for warning in section_result.warnings:
        ctx.record_warning(warning)

    with ctx.stage(
        "llm_report_metadata", "Polishing report summary with LLM", detail=runtime_detail
    ) as st:
        try:
            metadata_result = llm_processor.polish_report_metadata(
                report, section_transcripts=section_result.section_transcripts
            )
        except LlmProcessingError as error:
            ctx.record_warning(str(error))
            st.update(detail=_join_detail(runtime_detail, "fallback"))
            ctx.llm = LlmDiagnostics(
                model=model_name,
                report_status="fallback",
                report_latency_sec=section_elapsed_sec + st.elapsed_sec(),
                response_metadata=section_result.response_metadata,
            )
            return report
        metadata_elapsed_sec = st.elapsed_sec()
        st.update(detail=_metadata_detail(metadata_result, section_count=polish_plan.section_count))

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


def _metadata_detail(metadata_result: LlmReportMetadataResult, *, section_count: int) -> str | None:
    summary_count = len(metadata_result.summary)
    action_item_count = len(metadata_result.action_items)
    title_count = len(metadata_result.section_titles)
    title_update_count = title_count if title_count > 0 and title_count != section_count else 0
    parts = []
    if summary_count > 0:
        parts.append(_count(summary_count, "summary bullet"))
    if action_item_count > 0:
        parts.append(_count(action_item_count, "action item"))
    if title_update_count > 0:
        noun = "title" if title_update_count == 1 else "titles"
        parts.append(f"{title_update_count} {noun} updated")
    return _join_detail(*parts) or None


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _rtf(audio_sec: float, elapsed_sec: float) -> str:
    return f"RTF {format(round(audio_sec / elapsed_sec, 2), 'g')}x"


def _join_detail(*parts: str | None) -> str:
    return " | ".join(part for part in parts if part)


def _silent_reporter() -> StageReporter:
    return StageReporter(console=Console(quiet=True))


__all__ = [
    "INFERENCE_WINDOW_DURATION_SEC",
    "INFERENCE_WINDOW_OVERLAP_SEC",
    "ProcessArtifacts",
    "RunContext",
    "plan_inference_windows",
    "process_input",
]
