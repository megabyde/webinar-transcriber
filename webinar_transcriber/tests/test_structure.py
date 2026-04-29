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
from webinar_transcriber.structure.scoring import _derive_title
from webinar_transcriber.structure.sections import (
    _build_audio_sections,
    _first_words_title,
    _sections_from_block,
    _should_start_new_audio_section,
)


class TestAudioSectionBoundaries:
    @pytest.mark.parametrize(("gap_duration", "expected"), [(4.9, False), (5.0, True)])
    def test_starts_new_audio_section_for_boundary_conditions(
        self, gap_duration: float, expected: bool
    ) -> None:
        current_segments = [
            TranscriptSegment(
                id="segment-1", text="Current section text.", start_sec=0.0, end_sec=10.0
            )
        ]
        next_segment = TranscriptSegment(
            id="segment-2",
            text="More detail arrives immediately after that",
            start_sec=10.0 + gap_duration,
            end_sec=20.0 + gap_duration,
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

    def test_emits_empty_summary_and_action_items_without_llm(self) -> None:
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
        assert report.summary == []
        assert report.action_items == []

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
    def test_build_audio_sections_splits_when_gap_reaches_threshold(self) -> None:
        sections = _build_audio_sections([
            TranscriptSegment(
                id="segment-1", text="Long section opening.", start_sec=0.0, end_sec=140.0
            ),
            TranscriptSegment(
                id="segment-2",
                text="This should start a new section because the gap is large enough.",
                start_sec=148.0,
                end_sec=320.0,
            ),
        ])

        assert [(section.start_sec, section.end_sec) for section in sections] == [
            (0.0, 140.0),
            (148.0, 320.0),
        ]

    def test_does_not_split_before_gap_threshold_is_reached(self) -> None:
        current_segments = [
            TranscriptSegment(
                id="segment-1",
                text="Agenda review without a sentence break",
                start_sec=0.0,
                end_sec=10.0,
            )
        ]
        next_segment = TranscriptSegment(
            id="segment-2",
            text="More detail arrives immediately after that",
            start_sec=14.9,
            end_sec=30.0,
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

        sections = _sections_from_block(block, block_segments=[], next_section_index=1)

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
                        id="segment-1", text="So, okay, well.", start_sec=0.0, end_sec=5.0
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

    def test_derive_title_formats_local_path_stem(self) -> None:
        assert _derive_title("/recordings/weekly-sync.mp4") == "Weekly Sync"
