"""Tests for report structuring helpers."""

from webinar_transcriber.models import (
    AlignmentBlock,
    MediaAsset,
    MediaType,
    TranscriptionResult,
    TranscriptSegment,
)
from webinar_transcriber.structure import (
    _action_item_score,
    _audio_title_from_segments,
    _audio_title_score,
    _build_audio_sections,
    _build_summary,
    _extract_action_items,
    _fallback_summary,
    _segment_key,
    _summary_filler_penalty,
    _summary_repetition_penalty,
    _summary_start_penalty,
    _title_from_text,
    _title_from_words,
    build_report,
)

RU_SEND_FILE = "Пожалуйста, пришлите итоговый файл до пятницы."
RU_CHECK_NUMBERS = (
    "Не забудьте "  # noqa: RUF001
    "проверить "
    "финальные "
    "цифры перед "
    "созвоном."
)
RU_BUDGET_DISCUSSION = "Сегодня мы обсуждаем бюджет и план внедрения."


class TestBuildReport:
    def test_uses_alignment_block_title_hint_and_warnings(self) -> None:
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

    def test_falls_back_for_empty_text_and_dedupes_summary(self) -> None:
        report = build_report(
            MediaAsset(path="", media_type=MediaType.AUDIO, duration_sec=10.0),
            TranscriptionResult(
                segments=[
                    TranscriptSegment(id="segment-1", text="  ", start_sec=0.0, end_sec=1.0),
                    TranscriptSegment(
                        id="segment-2",
                        text="Repeat me.",
                        start_sec=1.0,
                        end_sec=2.0,
                    ),
                    TranscriptSegment(
                        id="segment-3",
                        text="Repeat me.",
                        start_sec=2.0,
                        end_sec=3.0,
                    ),
                    TranscriptSegment(
                        id="segment-4",
                        text="Please follow up.",
                        start_sec=3.0,
                        end_sec=4.0,
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

    def test_groups_audio_segments_into_larger_sections(self) -> None:
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

    def test_uses_more_informative_audio_section_title(self) -> None:
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

    def test_summary_skips_startup_chatter(self) -> None:
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

    def test_extracts_russian_action_items(self) -> None:
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

    def test_returns_no_audio_sections_for_blank_segments(self) -> None:
        report = build_report(
            MediaAsset(path="demo.wav", media_type=MediaType.AUDIO, duration_sec=30.0),
            TranscriptionResult(
                segments=[
                    TranscriptSegment(id="segment-1", text=" ", start_sec=0.0, end_sec=1.0),
                    TranscriptSegment(id="segment-2", text="", start_sec=1.0, end_sec=2.0),
                ],
            ),
        )

        assert report.sections == []


class TestAudioSectionHeuristics:
    def test_build_audio_sections_splits_when_target_duration_would_be_exceeded(self) -> None:
        sections = _build_audio_sections([
            TranscriptSegment(
                id="segment-1",
                text="Long section opening.",
                start_sec=0.0,
                end_sec=140.0,
            ),
            TranscriptSegment(
                id="segment-2",
                text="This should start a new section because it pushes duration too far.",
                start_sec=140.5,
                end_sec=320.0,
            ),
        ])

        assert [(section.start_sec, section.end_sec) for section in sections] == [
            (0.0, 140.0),
            (140.5, 320.0),
        ]

    def test_title_helpers_fall_back_for_empty_or_filler_only_text(self) -> None:
        filler_segments = [
            TranscriptSegment(id="segment-1", text="So, okay, well.", start_sec=0.0, end_sec=5.0),
            TranscriptSegment(
                id="segment-2",
                text="Just like okay right.",
                start_sec=5.0,
                end_sec=10.0,
            ),
        ]

        assert _title_from_text("   ", fallback="Fallback Title") == "Fallback Title"
        assert _audio_title_from_segments([], fallback="Fallback Title") == "Fallback Title"
        assert (
            _audio_title_from_segments(filler_segments, fallback="Fallback Title")
            == "Fallback Title"
        )
        assert _title_from_words([]) == ""
        assert _segment_key("   ") == ""

    def test_audio_title_score_covers_repetition_penalty_bands(self) -> None:
        heavy_repetition = TranscriptSegment(
            id="segment-1",
            text="Plan plan plan plan",
            start_sec=0.0,
            end_sec=4.0,
        )
        medium_repetition = TranscriptSegment(
            id="segment-2",
            text="Plan plan budget budget",
            start_sec=4.0,
            end_sec=8.0,
        )

        assert _audio_title_score(heavy_repetition) < _audio_title_score(medium_repetition)


class TestSummaryAndActionHeuristics:
    def test_build_summary_dedupes_repeated_high_value_segments(self) -> None:
        summary = _build_summary([
            TranscriptSegment(
                id="segment-1",
                text="We review staffing constraints and project delivery tradeoffs.",
                start_sec=120.0,
                end_sec=132.0,
            ),
            TranscriptSegment(
                id="segment-2",
                text="We review staffing constraints and project delivery tradeoffs.",
                start_sec=132.0,
                end_sec=144.0,
            ),
        ])

        assert summary == ["We review staffing constraints and project delivery tradeoffs."]

    def test_extract_action_items_dedupes_and_respects_limit(self) -> None:
        action_items = _extract_action_items([
            TranscriptSegment(id="segment-1", text="TODO", start_sec=120.0, end_sec=121.0),
            TranscriptSegment(
                id="segment-2",
                text="Please send the notes.",
                start_sec=121.0,
                end_sec=124.0,
            ),
            TranscriptSegment(
                id="segment-3",
                text="Please send the notes.",
                start_sec=124.0,
                end_sec=127.0,
            ),
            TranscriptSegment(
                id="segment-4",
                text="Please review the spreadsheet.",
                start_sec=127.0,
                end_sec=130.0,
            ),
            TranscriptSegment(
                id="segment-5",
                text="Please update the tracker.",
                start_sec=130.0,
                end_sec=133.0,
            ),
            TranscriptSegment(
                id="segment-6",
                text="Please share the recording.",
                start_sec=133.0,
                end_sec=136.0,
            ),
            TranscriptSegment(
                id="segment-7",
                text="Please check the final numbers.",
                start_sec=136.0,
                end_sec=139.0,
            ),
            TranscriptSegment(
                id="segment-8",
                text="Please remember the follow-up note.",
                start_sec=139.0,
                end_sec=142.0,
            ),
        ])

        assert action_items == [
            "Please send the notes.",
            "Please review the spreadsheet.",
            "Please update the tracker.",
            "Please share the recording.",
            "Please check the final numbers.",
        ]

    def test_structure_scoring_helpers_cover_penalty_branches(self) -> None:
        repetitive_segment = TranscriptSegment(
            id="segment-1",
            text="So so so so.",
            start_sec=0.0,
            end_sec=10.0,
        )
        filler_heavy_segment = TranscriptSegment(
            id="segment-2",
            text="Please chat audio microphone hello everyone.",
            start_sec=10.0,
            end_sec=14.0,
        )

        assert _audio_title_score(repetitive_segment) < 0
        assert _action_item_score(
            TranscriptSegment(id="segment-3", text="TODO", start_sec=0.0, end_sec=1.0)
        ) == -1.0
        assert _summary_start_penalty(200.0) == 0.0
        assert _summary_repetition_penalty(["plan"] * 6) == -3.0
        assert (
            _summary_repetition_penalty(
                ["plan", "budget", "timeline", "risk", "owner", "scope"]
            )
            == 0.0
        )
        assert _summary_filler_penalty(4, 10) == -2.0
        assert _summary_filler_penalty(2, 10) == -1.0
        assert _segment_key(
            filler_heavy_segment.text
        ) == "please chat audio microphone hello everyone"

    def test_action_item_score_penalizes_noise_and_missing_punctuation(self) -> None:
        score = _action_item_score(
            TranscriptSegment(
                id="segment-1",
                text="Please update the audio chat group",
                start_sec=10.0,
                end_sec=14.0,
            )
        )

        assert score < 0

    def test_summary_helpers_cover_medium_repetition_and_fallback_limit(self) -> None:
        summary = _fallback_summary([
            TranscriptSegment(id="segment-1", text="One", start_sec=0.0, end_sec=1.0),
            TranscriptSegment(id="segment-2", text="Two", start_sec=1.0, end_sec=2.0),
            TranscriptSegment(id="segment-3", text="Three", start_sec=2.0, end_sec=3.0),
            TranscriptSegment(id="segment-4", text="Four", start_sec=3.0, end_sec=4.0),
        ])

        assert (
            _summary_repetition_penalty(
                ["plan", "plan", "budget", "budget", "risk", "risk", "owner"]
            )
            == -1.5
        )
        assert summary == ["One", "Two", "Three"]
