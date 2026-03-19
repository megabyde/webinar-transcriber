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
