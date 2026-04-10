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
    _derive_title,
    _extract_action_items,
    _fallback_summary,
    _is_likely_interlude_text,
    _segment_key,
    _should_start_new_audio_section,
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
RU_REPETITIVE_SPEECH = (
    "Не в плане, что вы никого больше не трахаете, а в плане, "  # noqa: RUF001
    "что вы вырабатываете модель, которая для этой единственной "
    "уникальной женщины становится единственной и уникальной."
)


class TestAudioSectionBoundaries:
    def test_waits_for_sentence_boundary_before_soft_limit(self) -> None:
        current_segments = [
            TranscriptSegment(
                id="segment-1",
                text="This section has enough duration but no sentence boundary",
                start_sec=0.0,
                end_sec=130.0,
            )
        ]
        next_segment = TranscriptSegment(
            id="segment-2",
            text="More detail arrives immediately after that",
            start_sec=130.0,
            end_sec=310.0,
        )

        assert not _should_start_new_audio_section(current_segments, next_segment)

    def test_splits_at_hard_cap_without_sentence_boundary(self) -> None:
        current_segments = [
            TranscriptSegment(
                id="segment-1",
                text="This section keeps running without punctuation",
                start_sec=0.0,
                end_sec=350.0,
            )
        ]
        next_segment = TranscriptSegment(
            id="segment-2",
            text="More detail arrives immediately after that",
            start_sec=350.0,
            end_sec=610.0,
        )

        assert _should_start_new_audio_section(current_segments, next_segment)

    def test_splits_at_char_limit_with_sentence_boundary(self) -> None:
        current_segments = [
            TranscriptSegment(
                id="segment-1",
                text=f"{'a' * 3601}.",
                start_sec=0.0,
                end_sec=130.0,
            )
        ]
        next_segment = TranscriptSegment(
            id="segment-2",
            text="A short follow-up sentence.",
            start_sec=130.0,
            end_sec=150.0,
        )

        assert _should_start_new_audio_section(current_segments, next_segment)


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

    def test_splits_local_interlude_span_without_swallowing_surrounding_content(self) -> None:
        report = build_report(
            MediaAsset(path="demo-file.mp4", media_type=MediaType.VIDEO, duration_sec=56.0),
            TranscriptionResult(
                detected_language="ru",
                segments=[
                    TranscriptSegment(
                        id="segment-1",
                        text="Сегодня мы обсуждаем бюджет и план внедрения.",
                        start_sec=0.0,
                        end_sec=8.0,
                    ),
                    TranscriptSegment(
                        id="segment-2",
                        text="Субтитры сделал DimaTorzok",
                        start_sec=8.0,
                        end_sec=24.0,
                    ),
                    TranscriptSegment(
                        id="segment-3",
                        text="Lyrics interlude chorus verse.",
                        start_sec=24.0,
                        end_sec=40.0,
                    ),
                    TranscriptSegment(
                        id="segment-4",
                        text="Пожалуйста, пришлите итоговый файл до пятницы.",
                        start_sec=40.0,
                        end_sec=56.0,
                    ),
                ],
            ),
            alignment_blocks=[
                AlignmentBlock(
                    id="block-1",
                    start_sec=0.0,
                    end_sec=56.0,
                    transcript_segment_ids=[
                        "segment-1",
                        "segment-2",
                        "segment-3",
                        "segment-4",
                    ],
                    transcript_text="placeholder",
                    scene_id="scene-1",
                    frame_id="frame-1",
                )
            ],
        )

        assert [section.is_interlude for section in report.sections] == [False, True, False]
        assert report.sections[0].title != "Музыкальная пауза"
        assert report.sections[1].title == "Музыкальная пауза"
        assert report.sections[2].transcript_text == RU_SEND_FILE
        assert report.summary == [
            RU_BUDGET_DISCUSSION,
            RU_SEND_FILE,
        ]
        assert report.action_items == [RU_SEND_FILE]

    def test_renders_music_breaks_as_interludes_and_excludes_them_from_summary(self) -> None:
        report = build_report(
            MediaAsset(path="demo.wav", media_type=MediaType.AUDIO, duration_sec=180.0),
            TranscriptionResult(
                detected_language="ru",
                segments=[
                    TranscriptSegment(
                        id="segment-1",
                        text=("ля ля ля ля ля ля ля ля ля ля ля ля ля ля ля ля ля ля ля ля"),
                        start_sec=0.0,
                        end_sec=20.0,
                    ),
                    TranscriptSegment(
                        id="segment-2",
                        text=("ля ля ля ля ля ля ля ля ля ля ля ля ля ля ля ля ля ля ля ля"),
                        start_sec=20.0,
                        end_sec=45.0,
                    ),
                    TranscriptSegment(
                        id="segment-3",
                        text="Сегодня мы обсуждаем бюджет и план внедрения.",
                        start_sec=60.0,
                        end_sec=75.0,
                    ),
                    TranscriptSegment(
                        id="segment-4",
                        text="Пожалуйста, пришлите итоговый файл до пятницы.",
                        start_sec=75.0,
                        end_sec=90.0,
                    ),
                ],
            ),
        )

        assert report.sections[0].is_interlude
        assert report.sections[0].title == "Музыкальная пауза"
        assert "transcript.json" in report.sections[0].transcript_text
        assert report.summary == [
            "Сегодня мы обсуждаем бюджет и план внедрения.",
            "Пожалуйста, пришлите итоговый файл до пятницы.",
        ]
        assert report.action_items == ["Пожалуйста, пришлите итоговый файл до пятницы."]

    def test_renders_long_interlude_candidate(self) -> None:
        report = build_report(
            MediaAsset(path="demo.wav", media_type=MediaType.AUDIO, duration_sec=900.0),
            TranscriptionResult(
                detected_language="ru",
                segments=[
                    TranscriptSegment(
                        id="segment-1",
                        text="Субтитры сделал DimaTorzok",
                        start_sec=0.0,
                        end_sec=350.0,
                    ),
                    TranscriptSegment(
                        id="segment-2",
                        text="Lyrics interlude chorus verse.",
                        start_sec=350.0,
                        end_sec=700.0,
                    ),
                    TranscriptSegment(
                        id="segment-3",
                        text=RU_BUDGET_DISCUSSION,
                        start_sec=700.0,
                        end_sec=760.0,
                    ),
                ],
            ),
        )

        assert report.sections[0].is_interlude
        assert report.sections[0].title == "Музыкальная пауза"
        assert report.sections[1].transcript_text == RU_BUDGET_DISCUSSION

    def test_ignores_short_marker_only_interlude_candidate(self) -> None:
        report = build_report(
            MediaAsset(path="demo.wav", media_type=MediaType.AUDIO, duration_sec=120.0),
            TranscriptionResult(
                detected_language="ru",
                segments=[
                    TranscriptSegment(
                        id="segment-1",
                        text="Субтитры сделал DimaTorzok",
                        start_sec=0.0,
                        end_sec=18.0,
                    ),
                    TranscriptSegment(
                        id="segment-2",
                        text=RU_BUDGET_DISCUSSION,
                        start_sec=18.0,
                        end_sec=60.0,
                    ),
                ],
            ),
        )

        assert all(not section.is_interlude for section in report.sections)
        assert "Субтитры сделал DimaTorzok" in report.sections[0].transcript_text

    def test_does_not_mark_single_repetitive_speech_segment_as_interlude(self) -> None:
        report = build_report(
            MediaAsset(path="demo-file.mp4", media_type=MediaType.VIDEO, duration_sec=24.0),
            TranscriptionResult(
                detected_language="ru",
                segments=[
                    TranscriptSegment(
                        id="segment-1",
                        text="Сегодня мы обсуждаем бюджет и план внедрения.",
                        start_sec=0.0,
                        end_sec=8.0,
                    ),
                    TranscriptSegment(
                        id="segment-2",
                        text=RU_REPETITIVE_SPEECH,
                        start_sec=8.0,
                        end_sec=16.0,
                    ),
                    TranscriptSegment(
                        id="segment-3",
                        text="Пожалуйста, пришлите итоговый файл до пятницы.",
                        start_sec=16.0,
                        end_sec=24.0,
                    ),
                ],
            ),
            alignment_blocks=[
                AlignmentBlock(
                    id="block-1",
                    start_sec=0.0,
                    end_sec=24.0,
                    transcript_segment_ids=[
                        "segment-1",
                        "segment-2",
                        "segment-3",
                    ],
                    transcript_text="placeholder",
                    scene_id="scene-1",
                    frame_id="frame-1",
                )
            ],
        )

        assert all(not section.is_interlude for section in report.sections)
        assert len(report.sections) == 1
        assert "единственной и уникальной" in report.sections[0].transcript_text

    def test_does_not_mark_repetitive_technical_terms_as_interlude(self) -> None:
        assert not _is_likely_interlude_text(
            "HTTPS gRPC NCCL training pipeline gradient descent optimizer scheduler "
            "dataloader tensor cluster review metrics throughput gRPC NCCL "
            "optimizer scheduler tensor review"
        )


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

        filler_title = _audio_title_from_segments(
            filler_segments,
            fallback="Fallback Title",
        )

        assert _title_from_text("   ", fallback="Fallback Title") == "Fallback Title"
        assert _audio_title_from_segments([], fallback="Fallback Title") == "Fallback Title"
        assert filler_title == "Fallback Title"
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

    def test_extract_action_items_prefers_higher_scored_later_candidates(self) -> None:
        action_items = _extract_action_items([
            TranscriptSegment(
                id="segment-1",
                text="Please send the draft",
                start_sec=10.0,
                end_sec=13.0,
            ),
            TranscriptSegment(
                id="segment-2",
                text="Please review the spreadsheet",
                start_sec=123.0,
                end_sec=126.0,
            ),
            TranscriptSegment(
                id="segment-3",
                text="Please update the tracker",
                start_sec=126.0,
                end_sec=129.0,
            ),
            TranscriptSegment(
                id="segment-4",
                text="Please share the recording",
                start_sec=129.0,
                end_sec=132.0,
            ),
            TranscriptSegment(
                id="segment-5",
                text="Please check the notes",
                start_sec=132.0,
                end_sec=135.0,
            ),
            TranscriptSegment(
                id="segment-6",
                text="Please remember the final numbers.",
                start_sec=135.0,
                end_sec=138.0,
            ),
        ])

        assert action_items == [
            "Please review the spreadsheet",
            "Please update the tracker",
            "Please share the recording",
            "Please check the notes",
            "Please remember the final numbers.",
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

        todo_score = _action_item_score(
            TranscriptSegment(id="segment-3", text="TODO", start_sec=0.0, end_sec=1.0)
        )
        no_repetition_penalty = _summary_repetition_penalty([
            "plan",
            "budget",
            "timeline",
            "risk",
            "owner",
            "scope",
        ])
        filler_key = _segment_key(filler_heavy_segment.text)

        assert _audio_title_score(repetitive_segment) < 0
        assert todo_score == -1.0
        assert _summary_start_penalty(200.0) == 0.0
        assert _summary_repetition_penalty(["plan"] * 6) == -3.0
        assert no_repetition_penalty == 0.0
        assert _summary_filler_penalty(4, 10) == -2.0
        assert _summary_filler_penalty(2, 10) == -1.0
        assert filler_key == "please chat audio microphone hello everyone"

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
        repetition_penalty = _summary_repetition_penalty([
            "plan",
            "plan",
            "budget",
            "budget",
            "risk",
            "risk",
            "owner",
        ])

        assert repetition_penalty == -1.5
        assert summary == ["One", "Two", "Three"]

    def test_derive_title_handles_windows_paths(self) -> None:
        assert _derive_title(r"C:\recordings\weekly-sync.mp4") == "Weekly Sync"
