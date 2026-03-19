# Usage

The public CLI is intentionally small:

```bash
webinar-transcriber process INPUT
webinar-transcriber process INPUT --ocr
webinar-transcriber process INPUT --format docx
```

## Behavior

- The tool auto-detects whether `INPUT` is audio or video.
- `--ocr` is a video-only enhancement. On audio-only input it will emit a warning and continue.
- If no output directory is supplied, the tool writes a fresh run directory under `runs/`.
- JSON is always written; Markdown and DOCX follow the selected `--format`.

## Example

```bash
webinar-transcriber process webinar.mp4 --ocr --output-dir runs/demo
```

That run will:

- create `runs/demo/`
- normalize audio to `audio.wav`
- write `transcript.json`
- detect scenes and extract slide frames
- write `ocr.json` when OCR is enabled
- export `report.md`, `report.docx`, and `report.json`
