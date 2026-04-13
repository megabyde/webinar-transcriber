"""Tests for token-usage aggregation helpers."""

from webinar_transcriber.usage import merge_usage, merge_usage_into


class TestMergeUsage:
    def test_sums_matching_keys_across_dicts(self) -> None:
        merged = merge_usage(
            {"input_tokens": 10, "total_tokens": 14}, {"output_tokens": 4, "total_tokens": 4}, {}
        )

        assert merged == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 18}

    def test_merge_usage_into_mutates_target_totals(self) -> None:
        usage_totals = {"input_tokens": 10, "total_tokens": 14}

        merge_usage_into(usage_totals, {"output_tokens": 4, "total_tokens": 4})

        assert usage_totals == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 18}
