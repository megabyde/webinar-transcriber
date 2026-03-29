"""Helpers for aggregating token-usage accounting."""


def merge_usage(*usage_dicts: dict[str, int]) -> dict[str, int]:
    """Return one usage map with counts summed across all inputs."""
    merged: dict[str, int] = {}
    for usage in usage_dicts:
        for key, value in usage.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def merge_usage_into(target: dict[str, int], new_usage: dict[str, int]) -> None:
    """Mutate one usage map by adding another usage map into it."""
    for key, value in new_usage.items():
        target[key] = target.get(key, 0) + value
