"""Tests for compact human-facing label helpers."""

from webinar_transcriber.labels import optional_count_label


class TestOptionalCountLabel:
    def test_returns_blank_for_non_positive_counts(self) -> None:
        assert optional_count_label(0, "segment") == ""
        assert optional_count_label(-1, "segment") == ""
