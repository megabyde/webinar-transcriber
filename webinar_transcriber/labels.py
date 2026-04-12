"""Compact human-facing label helpers."""


def count_label(count: int, singular: str, *, plural: str | None = None) -> str:
    """Return a compact singular/plural count label."""
    resolved_plural = plural or f"{singular}s"
    label = singular if count == 1 else resolved_plural
    return f"{count} {label}"


def optional_count_label(count: int, singular: str, *, plural: str | None = None) -> str:
    """Return a count label only when the count is positive."""
    if count <= 0:
        return ""
    return count_label(count, singular, plural=plural)
