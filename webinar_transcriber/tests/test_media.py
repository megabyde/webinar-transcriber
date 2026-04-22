"""Tests for media probing helpers."""

from pathlib import Path

import pytest

from webinar_transcriber.media import (
    MediaProcessingError,
    _pyav_stream_has_attached_picture,
    _stream_duration_sec,
    open_output_media_container,
    probe_media,
)
from webinar_transcriber.models import AudioAsset, VideoAsset
from webinar_transcriber.tests.conftest import FakeContextContainer

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestProbeMedia:
    def test_reads_audio_metadata(self) -> None:
        asset = probe_media(FIXTURE_DIR / "sample-audio.mp3")

        assert isinstance(asset, AudioAsset)
        assert asset.duration_sec > 0
        assert asset.sample_rate

    def test_reads_video_metadata(self) -> None:
        asset = probe_media(FIXTURE_DIR / "sample-video.mp4")

        assert isinstance(asset, VideoAsset)
        assert asset.fps is not None
        assert asset.width is not None
        assert asset.height is not None

    def test_raises_when_pyav_reports_no_streams(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeContainer(FakeContextContainer):
            def __init__(self) -> None:
                self.streams: list[object] = []
                self.duration = None

        monkeypatch.setattr(
            "webinar_transcriber.media.av.open", lambda *_args, **_kwargs: FakeContainer()
        )

        with pytest.raises(MediaProcessingError, match="No audio or video stream found"):
            probe_media(FIXTURE_DIR / "sample-video.mp4")

    def test_wraps_open_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.media.av.open",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad open")),
        )

        with pytest.raises(MediaProcessingError, match=r"Could not open .* with PyAV: bad open"):
            probe_media(FIXTURE_DIR / "sample-video.mp4")


class TestAttachedPictureStream:
    def test_pyav_detects_attached_picture_flag(self) -> None:
        class FakeDisposition:
            attached_pic = 2

            def __and__(self, other: object) -> int:
                assert other == 2
                return 2

        stream = type("Stream", (), {"disposition": FakeDisposition()})()

        assert _pyav_stream_has_attached_picture(stream)

    def test_pyav_returns_false_when_attached_picture_flag_is_unavailable(self) -> None:
        assert not _pyav_stream_has_attached_picture(object())


class TestStreamDurationSec:
    def test_returns_none_without_duration_or_time_base(self) -> None:
        assert _stream_duration_sec(object()) is None


class TestOpenOutputMediaContainer:
    def test_wraps_open_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            "webinar_transcriber.media.av.open",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad open")),
        )

        with pytest.raises(MediaProcessingError, match="wrapped bad open"):
            open_output_media_container(
                tmp_path / "out.wav",
                error_message="wrapped {error}",
            ).__enter__()
