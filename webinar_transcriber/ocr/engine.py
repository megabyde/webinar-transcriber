"""Tesseract OCR helpers."""

from pathlib import Path

import pytesseract
from PIL import Image
from pytesseract import Output

from webinar_transcriber.models import OcrResult, SlideFrame

LANGUAGE_MAP = {
    "en": "eng",
    "ru": "rus",
}


def resolve_tesseract_languages(detected_language: str | None) -> str:
    """Map the detected ASR language to a Tesseract language string."""
    if detected_language is None:
        return "eng+rus"
    return LANGUAGE_MAP.get(detected_language.lower(), "eng+rus")


def extract_ocr_results(
    slide_frames: list[SlideFrame],
    *,
    detected_language: str | None,
) -> list[OcrResult]:
    """Run OCR on the provided slide frame images."""
    language = resolve_tesseract_languages(detected_language)
    results: list[OcrResult] = []

    for frame in slide_frames:
        image = Image.open(Path(frame.image_path))
        data = pytesseract.image_to_data(
            image,
            lang=language,
            output_type=Output.DICT,
        )
        text_parts = [part.strip() for part in data["text"] if part.strip()]
        confidences = [int(raw) for raw in data["conf"] if raw != "-1"]
        confidence = (sum(confidences) / len(confidences) / 100) if confidences else None
        results.append(
            OcrResult(
                frame_id=frame.id,
                text=" ".join(text_parts),
                confidence=confidence,
                language=language,
            )
        )

    return results
