"""Shared test helpers for webinar_transcriber/tests."""

from collections.abc import Callable
from typing import Self

import pytest


class FakeTorch:
    @staticmethod
    def from_numpy(arr):
        return arr


class FakeContextContainer:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


@pytest.fixture
def fake_silero_import_module() -> Callable[..., Callable[[str], object]]:
    def build(
        *,
        iterator_cls: type | None = None,
        get_speech_timestamps_fn: Callable[..., object] | None = None,
    ) -> Callable[[str], object]:
        class FakeSilero:
            VADIterator: type | None = None
            get_speech_timestamps: Callable[..., object] | None = None

            @staticmethod
            def load_silero_vad():
                return object()

        if iterator_cls is not None:  # pragma: no cover - fixture option for narrow VAD tests
            FakeSilero.VADIterator = iterator_cls
        if get_speech_timestamps_fn is not None:
            FakeSilero.get_speech_timestamps = staticmethod(get_speech_timestamps_fn)

        def fake_import_module(name: str) -> object:
            if name == "silero_vad":
                return FakeSilero
            if name == "torch":
                return FakeTorch
            raise ImportError(name)  # pragma: no cover - mirrors importlib for unexpected modules

        return fake_import_module

    return build
