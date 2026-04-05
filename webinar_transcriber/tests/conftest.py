"""Shared test helpers for webinar_transcriber/tests."""

from collections.abc import Callable

import pytest


class FakeTorch:
    @staticmethod
    def from_numpy(arr):
        return arr


@pytest.fixture
def fake_silero_import_module() -> Callable[..., Callable[[str], object]]:
    def build(*, iterator_cls: type | None = None) -> Callable[[str], object]:
        class FakeSilero:
            VADIterator: type | None = None

            @staticmethod
            def load_silero_vad():
                return object()

        if iterator_cls is not None:
            FakeSilero.VADIterator = iterator_cls

        def fake_import_module(name: str) -> object:
            if name == "silero_vad":
                return FakeSilero
            if name == "torch":
                return FakeTorch
            raise ImportError(name)

        return fake_import_module

    return build
