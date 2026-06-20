"""Tests for report structuring helpers."""

import pytest

from webinar_transcriber.models import (
    AudioAsset,
    Scene,
    TranscriptionResult,
    TranscriptSegment,
    VideoAsset,
)
from webinar_transcriber.structure import (
    build_audio_sections,
    build_report,
    build_video_sections,
    derive_title,
    should_start_new_audio_section,
    title_from_path,
)


class TestBuildVideoSections:
    def test_assigns_segments_to_scenes_by_midpoint(self) -> None:
        sections = build_video_sections(
            segments=[
                TranscriptSegment(id="seg-1", text="Intro", start_sec=0.0, end_sec=0.8),
                TranscriptSegment(id="seg-2", text="Demo", start_sec=1.2, end_sec=1.8),
            ],
            scenes=[
                Scene(id="scene-1", start_sec=0.0, end_sec=1.0, image_path="frames/scene-1.png"),
                Scene(id="scene-2", start_sec=1.0, end_sec=2.0, image_path="frames/scene-2.png"),
            ],
        )

        assert [section.transcript_text for section in sections] == ["Intro", "Demo"]
        assert [section.id for section in sections] == ["section-1", "section-2"]
        assert sections[0].image_path == "frames/scene-1.png"
        assert sections[1].image_path == "frames/scene-2.png"

    def test_returns_empty_list_when_there_are_no_scenes(self) -> None:
        sections = build_video_sections(
            segments=[TranscriptSegment(id="seg-1", text="Intro", start_sec=0.0, end_sec=0.8)],
            scenes=[],
        )

        assert sections == []

    def test_keeps_scene_only_sections_without_segments(self) -> None:
        sections = build_video_sections(
            segments=[TranscriptSegment(id="seg-1", text="Demo", start_sec=1.2, end_sec=1.8)],
            scenes=[
                Scene(id="scene-1", start_sec=0.0, end_sec=1.0, image_path="frames/scene-1.png"),
                Scene(id="scene-2", start_sec=1.0, end_sec=2.0),
            ],
        )

        assert sections[0].id == "section-1"
        assert sections[0].title == "Slide 1"
        assert (sections[0].start_sec, sections[0].end_sec) == (0.0, 1.0)
        assert sections[0].transcript_text == ""
        assert sections[0].image_path == "frames/scene-1.png"
        assert sections[1].id == "section-2"
        assert sections[1].title == "Demo"
        assert (sections[1].start_sec, sections[1].end_sec) == (1.2, 1.8)

    def test_blank_segments_fall_back_to_scene_bounds(self) -> None:
        sections = build_video_sections(
            segments=[TranscriptSegment(id="seg-1", text="   ", start_sec=0.2, end_sec=0.8)],
            scenes=[Scene(id="scene-1", start_sec=0.0, end_sec=1.0)],
        )

        assert len(sections) == 1
        assert sections[0].title == "Slide 1"
        assert (sections[0].start_sec, sections[0].end_sec) == (0.0, 1.0)
        assert sections[0].transcript_text == ""


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

        assert should_start_new_audio_section(current_segments, next_segment) == expected


class TestBuildReport:
    def test_aligns_video_sections_and_keeps_warnings(self) -> None:
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
            scenes=[
                Scene(id="scene-1", start_sec=0.0, end_sec=12.0, image_path="frames/scene-1.png")
            ],
        )

        assert report.title == "Demo File"
        assert report.sections[0].title == "Agenda overview"
        assert report.sections[0].image_path == "frames/scene-1.png"

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
        assert report.sections[0].title == "Repeat me"
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
        assert report.sections[0].title == "So, well, okay, let's get started"

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

    def test_groups_adjacent_speaker_segments_into_paragraphs(self) -> None:
        report = build_report(
            AudioAsset(path="demo.wav", duration_sec=30.0),
            TranscriptionResult(
                segments=[
                    TranscriptSegment(
                        id="segment-1",
                        text="Agenda review.",
                        start_sec=0.0,
                        end_sec=2.0,
                        speaker="S1",
                    ),
                    TranscriptSegment(
                        id="segment-2",
                        text="Project status.",
                        start_sec=2.0,
                        end_sec=4.0,
                        speaker="S1",
                    ),
                    TranscriptSegment(
                        id="segment-3", text="Next steps.", start_sec=4.0, end_sec=6.0, speaker="S2"
                    ),
                ]
            ),
        )

        assert report.sections[0].transcript_text == (
            "**S1:** Agenda review.\n\nProject status.\n\n**S2:** Next steps."
        )


class TestAudioSectionHeuristics:
    def test_build_audio_sections_splits_when_gap_reaches_threshold(self) -> None:
        sections = build_audio_sections([
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
        assert [section.id for section in sections] == ["section-1", "section-2"]

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

        assert not should_start_new_audio_section(current_segments, next_segment)

    def test_numbers_audio_fallback_titles_by_position(self) -> None:
        sections = build_audio_sections([
            TranscriptSegment(id="segment-1", text="...", start_sec=0.0, end_sec=10.0),
            TranscriptSegment(id="segment-2", text="...", start_sec=20.0, end_sec=30.0),
        ])

        assert [section.title for section in sections] == ["Section 1", "Section 2"]

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
                "So, okay, well",
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
    def test_derive_title_from_transcript_text(
        self, segments: list[TranscriptSegment], fallback: str, expected: str
    ) -> None:
        text = next((segment.text for segment in segments if segment.text.strip()), "")

        assert derive_title(text, fallback=fallback, ellipsis=True) == expected

    def test_title_from_path_formats_local_path_stem(self) -> None:
        assert title_from_path("/recordings/weekly-sync.mp4") == "Weekly Sync"
