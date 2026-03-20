"""Tests for OCR helpers."""

from pathlib import Path

from webinar_transcriber.models import SlideFrame
from webinar_transcriber.ocr import extract_ocr_results, resolve_tesseract_languages

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_resolve_tesseract_languages_defaults_to_english_and_russian() -> None:
    assert resolve_tesseract_languages(None) == "eng+rus"
    assert resolve_tesseract_languages("en") == "eng"
    assert resolve_tesseract_languages("ru") == "rus"


def test_extract_ocr_results_reads_fixture_text() -> None:
    progress_ticks: list[int] = []
    results = extract_ocr_results(
        [
            SlideFrame(
                id="frame-1",
                scene_id="scene-1",
                image_path=str(FIXTURE_DIR / "ocr-slide.png"),
                timestamp_sec=0.0,
            )
        ],
        detected_language="en",
        progress_callback=lambda: progress_ticks.append(1),
    )

    assert results[0].text
    assert "Agenda" in results[0].text or "AGENDA" in results[0].text
    assert len(progress_ticks) == 1
