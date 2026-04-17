"""Tests for report structuring helpers."""

import pytest

from webinar_transcriber.models import (
    AlignmentBlock,
    AudioAsset,
    TranscriptionResult,
    TranscriptSegment,
    VideoAsset,
)
from webinar_transcriber.structure import build_report
from webinar_transcriber.structure.scoring import (
    _build_summary,
    _derive_title,
    _extract_action_items,
    _fallback_summary,
    _segment_key,
    _summary_score,
)
from webinar_transcriber.structure.sections import (
    _build_audio_sections,
    _first_words_title,
    _sections_from_block,
    _should_start_new_audio_section,
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


class TestAudioSectionBoundaries:
    @pytest.mark.parametrize(
        ("current_text", "current_end_sec", "next_end_sec", "expected"),
        [
            ("This section has enough duration but no sentence boundary", 130.0, 310.0, False),
            ("This section keeps running without punctuation", 350.0, 610.0, True),
            (f"{'a' * 3601}.", 130.0, 150.0, True),
        ],
    )
    def test_starts_new_audio_section_for_boundary_conditions(
        self, current_text: str, current_end_sec: float, next_end_sec: float, expected: bool
    ) -> None:
        current_segments = [
            TranscriptSegment(
                id="segment-1", text=current_text, start_sec=0.0, end_sec=current_end_sec
            )
        ]
        next_segment = TranscriptSegment(
            id="segment-2",
            text="More detail arrives immediately after that",
            start_sec=current_end_sec,
            end_sec=next_end_sec,
        )

        assert _should_start_new_audio_section(current_segments, next_segment) == expected


class TestBuildReport:
    def test_uses_alignment_block_text_and_warnings(self) -> None:
        report = build_report(
            VideoAsset(path="demo-file.mp4", duration_sec=12.0),
            TranscriptionResult(
                detected_language="en",
                segments=[
                    TranscriptSegment(
                        id="segment-1", text="Agenda overview", start_sec=0.0, end_sec=2.0
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
                )
            ],
            warnings=["low confidence"],
        )

        assert report.title == "Demo File"
        assert report.sections[0].title == "Agenda overview"
        assert report.sections[0].frame_id == "frame-1"
        assert report.warnings == ["low confidence"]

    def test_falls_back_for_empty_text_and_dedupes_summary(self) -> None:
        report = build_report(
            AudioAsset(path="", duration_sec=10.0),
            TranscriptionResult(
                segments=[
                    TranscriptSegment(id="segment-1", text="  ", start_sec=0.0, end_sec=1.0),
                    TranscriptSegment(
                        id="segment-2", text="Repeat me.", start_sec=1.0, end_sec=2.0
                    ),
                    TranscriptSegment(
                        id="segment-3", text="Repeat me.", start_sec=2.0, end_sec=3.0
                    ),
                    TranscriptSegment(
                        id="segment-4", text="Please follow up.", start_sec=3.0, end_sec=4.0
                    ),
                ]
            ),
        )

        assert report.title == "Transcription Report"
        assert len(report.sections) == 1
        assert report.sections[0].title == "Repeat me."
        assert "Please follow up." in report.sections[0].transcript_text
        assert report.summary == ["Repeat me.", "Please follow up."]
        assert report.action_items == ["Please follow up."]

    def test_groups_audio_segments_into_larger_sections(self) -> None:
        report = build_report(
            AudioAsset(path="demo.wav", duration_sec=80.0),
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
                ]
            ),
        )

        assert len(report.sections) == 2
        assert report.sections[0].start_sec == 0.0
        assert report.sections[0].end_sec == 20.0
        assert "Budget discussion" in report.sections[0].transcript_text
        assert report.sections[1].start_sec == 30.0
        assert report.sections[1].end_sec == 39.0

    def test_uses_first_words_audio_section_title(self) -> None:
        report = build_report(
            AudioAsset(path="demo.wav", duration_sec=260.0),
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
                ]
            ),
        )

        assert len(report.sections) == 1
        assert report.sections[0].title == "So, well, okay, let's get started."

    def test_summary_skips_startup_chatter(self) -> None:
        report = build_report(
            AudioAsset(path="demo.wav", duration_sec=260.0),
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
                ]
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
            AudioAsset(path="demo.wav", duration_sec=120.0),
            TranscriptionResult(
                segments=[
                    TranscriptSegment(
                        id="segment-1", text=RU_SEND_FILE, start_sec=65.0, end_sec=72.0
                    ),
                    TranscriptSegment(
                        id="segment-2", text=RU_CHECK_NUMBERS, start_sec=72.0, end_sec=79.0
                    ),
                    TranscriptSegment(
                        id="segment-3", text=RU_BUDGET_DISCUSSION, start_sec=79.0, end_sec=88.0
                    ),
                ]
            ),
        )

        assert report.action_items == [RU_SEND_FILE, RU_CHECK_NUMBERS]

    def test_returns_no_audio_sections_for_blank_segments(self) -> None:
        report = build_report(
            AudioAsset(path="demo.wav", duration_sec=30.0),
            TranscriptionResult(
                segments=[
                    TranscriptSegment(id="segment-1", text=" ", start_sec=0.0, end_sec=1.0),
                    TranscriptSegment(id="segment-2", text="", start_sec=1.0, end_sec=2.0),
                ]
            ),
        )

        assert report.sections == []


class TestAudioSectionHeuristics:
    def test_build_audio_sections_splits_when_target_duration_would_be_exceeded(self) -> None:
        sections = _build_audio_sections([
            TranscriptSegment(
                id="segment-1", text="Long section opening.", start_sec=0.0, end_sec=140.0
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

    def test_does_not_split_before_budget_or_gap_threshold_is_reached(self) -> None:
        current_segments = [
            TranscriptSegment(
                id="segment-1",
                text="Agenda review without a sentence break",
                start_sec=0.0,
                end_sec=140.0,
            )
        ]
        next_segment = TranscriptSegment(
            id="segment-2",
            text="More detail arrives immediately after that",
            start_sec=140.2,
            end_sec=200.0,
        )

        assert not _should_start_new_audio_section(current_segments, next_segment)

    def test_sections_from_block_uses_block_text_when_segment_ids_are_missing(self) -> None:
        block = AlignmentBlock(
            id="block-1",
            start_sec=0.0,
            end_sec=10.0,
            transcript_segment_ids=["missing-segment"],
            transcript_text="Fallback block text",
            scene_id="scene-1",
            frame_id="frame-1",
        )

        sections = _sections_from_block(
            block,
            block_segments=[],
            next_section_index=1,
        )

        assert len(sections) == 1
        assert sections[0].title == "Fallback block text"
        assert sections[0].frame_id == "frame-1"

    @pytest.mark.parametrize(
        ("segments", "fallback", "expected"),
        [
            ([], "Fallback Title", "Fallback Title"),
            (
                [TranscriptSegment(id="segment-1", text="   ", start_sec=0.0, end_sec=5.0)],
                "Fallback Title",
                "Fallback Title",
            ),
            (
                [
                    TranscriptSegment(
                        id="segment-1",
                        text="So, okay, well.",
                        start_sec=0.0,
                        end_sec=5.0,
                    )
                ],
                "Fallback Title",
                "So, okay, well.",
            ),
            (
                [
                    TranscriptSegment(
                        id="segment-1",
                        text="One two three four five six seven",
                        start_sec=0.0,
                        end_sec=5.0,
                    )
                ],
                "Fallback Title",
                "One two three four five six…",
            ),
        ],
    )
    def test_first_words_title(
        self, segments: list[TranscriptSegment], fallback: str, expected: str
    ) -> None:
        assert _first_words_title(segments, fallback=fallback) == expected
        assert _segment_key("   ") == ""


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
                id="segment-2", text="Please send the notes.", start_sec=121.0, end_sec=124.0
            ),
            TranscriptSegment(
                id="segment-3", text="Please send the notes.", start_sec=124.0, end_sec=127.0
            ),
            TranscriptSegment(
                id="segment-4",
                text="Please review the spreadsheet.",
                start_sec=127.0,
                end_sec=130.0,
            ),
            TranscriptSegment(
                id="segment-5", text="Please update the tracker.", start_sec=130.0, end_sec=133.0
            ),
            TranscriptSegment(
                id="segment-6", text="Please share the recording.", start_sec=133.0, end_sec=136.0
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

    def test_extract_action_items_keeps_first_matching_candidates_in_transcript_order(self) -> None:
        action_items = _extract_action_items([
            TranscriptSegment(
                id="segment-1", text="Please send the draft", start_sec=10.0, end_sec=13.0
            ),
            TranscriptSegment(
                id="segment-2", text="Please review the spreadsheet", start_sec=123.0, end_sec=126.0
            ),
            TranscriptSegment(
                id="segment-3", text="Please update the tracker", start_sec=126.0, end_sec=129.0
            ),
            TranscriptSegment(
                id="segment-4", text="Please share the recording", start_sec=129.0, end_sec=132.0
            ),
            TranscriptSegment(
                id="segment-5", text="Please check the notes", start_sec=132.0, end_sec=135.0
            ),
            TranscriptSegment(
                id="segment-6",
                text="Please remember the final numbers.",
                start_sec=135.0,
                end_sec=138.0,
            ),
        ])

        assert action_items == [
            "Please send the draft",
            "Please review the spreadsheet",
            "Please update the tracker",
            "Please share the recording",
            "Please check the notes",
        ]

    def test_structure_scoring_helpers_cover_reduced_scoring_paths(self) -> None:
        filler_heavy_segment = TranscriptSegment(
            id="segment-2",
            text="Please chat audio microphone hello everyone.",
            start_sec=10.0,
            end_sec=14.0,
        )

        filler_key = _segment_key(filler_heavy_segment.text)

        assert filler_key == "please chat audio microphone hello everyone"

    def test_action_items_skip_noise_even_with_action_cue(self) -> None:
        action_items = _extract_action_items([
            TranscriptSegment(
                id="segment-1",
                text="Please update the audio chat group",
                start_sec=10.0,
                end_sec=14.0,
            )
        ])

        assert action_items == []

    def test_action_items_preserve_transcript_order_after_cue_filtering(self) -> None:
        action_items = _extract_action_items([
            TranscriptSegment(
                id="segment-1",
                text="Please send the draft by Friday.",
                start_sec=120.0,
                end_sec=124.0,
            ),
            TranscriptSegment(
                id="segment-2",
                text="Please update the tracker after the review.",
                start_sec=10.0,
                end_sec=14.0,
            ),
        ])

        assert action_items == [
            "Please send the draft by Friday.",
            "Please update the tracker after the review.",
        ]

    def test_summary_score_penalizes_repetitive_text(self) -> None:
        repetitive_score = _summary_score(
            TranscriptSegment(
                id="segment-1",
                text="Plan plan plan plan budget budget budget budget risk risk risk risk.",
                start_sec=120.0,
                end_sec=132.0,
            )
        )
        informative_score = _summary_score(
            TranscriptSegment(
                id="segment-2",
                text=(
                    "Plan budget risks staffing delivery owners milestones blockers "
                    "timeline rollout metrics approvals."
                ),
                start_sec=120.0,
                end_sec=132.0,
            )
        )

        assert repetitive_score < informative_score

    def test_fallback_summary_keeps_limit(self) -> None:
        summary = _fallback_summary([
            TranscriptSegment(id="segment-1", text="One", start_sec=0.0, end_sec=1.0),
            TranscriptSegment(id="segment-2", text="Two", start_sec=1.0, end_sec=2.0),
            TranscriptSegment(id="segment-3", text="Three", start_sec=2.0, end_sec=3.0),
            TranscriptSegment(id="segment-4", text="Four", start_sec=3.0, end_sec=4.0),
        ])

        assert summary == ["One", "Two", "Three"]

    def test_derive_title_formats_local_path_stem(self) -> None:
        assert _derive_title("/recordings/weekly-sync.mp4") == "Weekly Sync"
