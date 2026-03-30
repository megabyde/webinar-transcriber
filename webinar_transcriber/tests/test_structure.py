"""Tests for report structuring helpers."""

from webinar_transcriber.models import (
    AlignmentBlock,
    MediaAsset,
    MediaType,
    TranscriptionResult,
    TranscriptSegment,
)
from webinar_transcriber.structure import build_report

RU_SEND_FILE = "Пожалуйста, пришлите итоговый файл до пятницы."
RU_CHECK_NUMBERS = (
    "Не забудьте "  # noqa: RUF001
    "проверить "
    "финальные "
    "цифры перед "
    "созвоном."
)
RU_BUDGET_DISCUSSION = "Сегодня мы обсуждаем бюджет и план внедрения."


def test_build_report_uses_alignment_block_title_hint_and_warnings() -> None:
    report = build_report(
        MediaAsset(path="demo-file.mp4", media_type=MediaType.VIDEO, duration_sec=12.0),
        TranscriptionResult(
            detected_language="en",
            segments=[
                TranscriptSegment(
                    id="segment-1",
                    text="Agenda overview",
                    start_sec=0.0,
                    end_sec=2.0,
                )
            ],
        ),
        alignment_blocks=[
            AlignmentBlock(
                id="block-1",
                start_sec=0.0,
                end_sec=12.0,
                transcript_segment_ids=["segment-1"],
                transcript_text="Agenda overview",
                scene_id="scene-1",
                frame_id="frame-1",
                title_hint="Slide Title",
            )
        ],
        warnings=["low confidence"],
    )

    assert report.title == "Demo File"
    assert report.sections[0].title == "Slide Title"
    assert report.sections[0].frame_id == "frame-1"
    assert report.warnings == ["low confidence"]


def test_build_report_falls_back_for_empty_text_and_dedupes_summary() -> None:
    report = build_report(
        MediaAsset(path="", media_type=MediaType.AUDIO, duration_sec=10.0),
        TranscriptionResult(
            segments=[
                TranscriptSegment(id="segment-1", text="  ", start_sec=0.0, end_sec=1.0),
                TranscriptSegment(id="segment-2", text="Repeat me.", start_sec=1.0, end_sec=2.0),
                TranscriptSegment(id="segment-3", text="Repeat me.", start_sec=2.0, end_sec=3.0),
                TranscriptSegment(
                    id="segment-4", text="Please follow up.", start_sec=3.0, end_sec=4.0
                ),
            ],
        ),
    )

    assert report.title == "Transcription Report"
    assert len(report.sections) == 1
    assert report.sections[0].title == "Repeat me"
    assert "Please follow up." in report.sections[0].transcript_text
    assert report.summary == ["Repeat me.", "Please follow up."]
    assert report.action_items == ["Please follow up."]


def test_build_report_groups_audio_segments_into_larger_sections() -> None:
    report = build_report(
        MediaAsset(path="demo.wav", media_type=MediaType.AUDIO, duration_sec=80.0),
        TranscriptionResult(
            segments=[
                TranscriptSegment(
                    id="segment-1",
                    text="Agenda review and project status update.",
                    start_sec=0.0,
                    end_sec=10.0,
                ),
                TranscriptSegment(
                    id="segment-2",
                    text="Budget discussion and delivery timeline review.",
                    start_sec=10.5,
                    end_sec=20.0,
                ),
                TranscriptSegment(
                    id="segment-3",
                    text="Open questions from the audience and follow-up items.",
                    start_sec=30.0,
                    end_sec=39.0,
                ),
            ],
        ),
    )

    assert len(report.sections) == 2
    assert report.sections[0].start_sec == 0.0
    assert report.sections[0].end_sec == 20.0
    assert "Budget discussion" in report.sections[0].transcript_text
    assert report.sections[1].start_sec == 30.0
    assert report.sections[1].end_sec == 39.0


def test_build_report_uses_more_informative_audio_section_title() -> None:
    report = build_report(
        MediaAsset(path="demo.wav", media_type=MediaType.AUDIO, duration_sec=260.0),
        TranscriptionResult(
            segments=[
                TranscriptSegment(
                    id="segment-1",
                    text="So, well, okay, let's get started.",
                    start_sec=0.0,
                    end_sec=10.0,
                ),
                TranscriptSegment(
                    id="segment-2",
                    text="Today we review budget negotiations and project delivery timelines.",
                    start_sec=10.0,
                    end_sec=22.0,
                ),
                TranscriptSegment(
                    id="segment-3",
                    text="First we examine the risks and then the approval options.",
                    start_sec=22.0,
                    end_sec=34.0,
                ),
            ],
        ),
    )

    assert len(report.sections) == 1
    assert report.sections[0].title == "Today we review budget negotiations and"


def test_build_report_summary_skips_startup_chatter() -> None:
    report = build_report(
        MediaAsset(path="demo.wav", media_type=MediaType.AUDIO, duration_sec=260.0),
        TranscriptionResult(
            segments=[
                TranscriptSegment(
                    id="segment-1",
                    text="Hello everyone, can you hear me clearly?",
                    start_sec=0.0,
                    end_sec=6.0,
                ),
                TranscriptSegment(
                    id="segment-2",
                    text="Please drop a note in the chat if the audio is working.",
                    start_sec=6.0,
                    end_sec=14.0,
                ),
                TranscriptSegment(
                    id="segment-3",
                    text="Today we review budget negotiations and project delivery timelines.",
                    start_sec=120.0,
                    end_sec=132.0,
                ),
                TranscriptSegment(
                    id="segment-4",
                    text="We compare approval risks, fallback plans, and delivery tradeoffs.",
                    start_sec=132.0,
                    end_sec=144.0,
                ),
                TranscriptSegment(
                    id="segment-5",
                    text="The final section covers next-quarter staffing constraints.",
                    start_sec=144.0,
                    end_sec=154.0,
                ),
            ],
        ),
    )

    expected_summary = [
        "Today we review budget negotiations and project delivery timelines.",
        "We compare approval risks, fallback plans, and delivery tradeoffs.",
        "The final section covers next-quarter staffing constraints.",
    ]

    assert report.summary == expected_summary


def test_build_report_extracts_russian_action_items() -> None:
    report = build_report(
        MediaAsset(path="demo.wav", media_type=MediaType.AUDIO, duration_sec=120.0),
        TranscriptionResult(
            segments=[
                TranscriptSegment(
                    id="segment-1",
                    text=RU_SEND_FILE,
                    start_sec=65.0,
                    end_sec=72.0,
                ),
                TranscriptSegment(
                    id="segment-2",
                    text=RU_CHECK_NUMBERS,
                    start_sec=72.0,
                    end_sec=79.0,
                ),
                TranscriptSegment(
                    id="segment-3",
                    text=RU_BUDGET_DISCUSSION,
                    start_sec=79.0,
                    end_sec=88.0,
                ),
            ],
        ),
    )

    assert report.action_items == [RU_SEND_FILE, RU_CHECK_NUMBERS]
