# Architecture

`webinar-transcriber` is structured around a single `process` command with an optional `--ocr`
enhancement path.

## Pipeline

1. Probe input media with `ffprobe`.
2. Normalize audio with `ffmpeg`.
3. Transcribe with `faster-whisper`.
4. For video input, detect scenes and extract representative frames.
5. Optionally run OCR on frames.
6. Align transcript content to audio or slide sections.
7. Structure notes and export Markdown, DOCX, and JSON artifacts.

## Runtime Behavior

- Audio inputs flow through probe, extraction, ASR, structuring, and export.
- Video inputs add scene detection, frame extraction, and time-based alignment.
- `--ocr` overlays Tesseract OCR on the video path and blends OCR text with timing cues.
- Every invocation writes a fresh run directory and diagnostics JSON.

## Package Shape

The package stays intentionally flat. Top-level modules own the core flow, with lightweight
subpackages only where the implementation naturally grows, such as video, OCR, and export code.
