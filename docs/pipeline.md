# Pipeline

This document describes the deterministic stages the CLI runs from input file to final artifacts.
For install and typical usage, see [README.md](../README.md).

1. Create an isolated run directory.
   - Each input gets its own `runs/<timestamp>_<basename>/` directory unless you pass a single
     `--output-dir`.
   - Existing output directories are refused instead of overwritten, so reruns do not silently mix
     old and new artifacts.
1. Probe the media.
   - The probe records whether the input is audio or video, its duration, and the streams available
     to the rest of the pipeline.
   - This data is written early to `metadata.json`, and later stages use the probed duration for
     progress, diagnostics, and report timing.
1. Prepare deterministic transcription audio.
   - PyAV decodes the input into temporary mono, `16 kHz`, `16-bit PCM WAV` audio.
   - Downstream speech detection, ASR, and diarization all consume this normalized audio contract.
   - The temporary WAV normally stays outside the run directory; `--keep-audio` saves a compressed
     `transcription-audio.mp3` copy after transcription.
1. Load the local ASR runtime.
   - `pywhispercpp` loads the selected `whisper.cpp` model identifier or local GGML model path.
   - The backend, model name, thread count, and native whisper.cpp logs are recorded in the run
     directory so performance and failures can be inspected after the run.
1. Detect speech regions.
   - The bundled Silero VAD ONNX model runs locally through `sherpa-onnx`.
   - Speech ranges are padded, overlapping padded ranges are merged, and the result is written to
     `asr/speech_regions.json`.
   - If VAD cannot run on the host, the pipeline records a warning and falls back to treating the
     whole normalized audio file as speech.
1. Plan ASR windows.
   - Speech regions are converted into bounded Whisper decode windows.
   - Long regions are split into windows of about 28 seconds with a small overlap at boundaries, so
     Whisper gets context without decoding the entire file at once.
1. Transcribe windows locally.
   - Each window is decoded by `whisper.cpp` and mapped back onto the original media timeline.
   - The raw per-window result is written to `asr/decoded_windows.json`; this is useful when
     debugging boundary behavior before reconciliation.
1. Reconcile raw windows into a transcript.
   - Overlapping window boundaries are deduplicated by comparing text prefixes and suffixes.
   - Empty or invalid decoded segments are discarded.
   - The reconciled transcript is the source for `transcript.json` and for optional speaker
     assignment.
1. Optionally diarize speakers.
   - `--diarize` runs local `sherpa-onnx` speaker segmentation and embedding models against the same
     normalized audio.
   - Speaker turns are normalized to anonymous labels ordered by first appearance (`S1`, `S2`, and
     so on) and written to `diarization.json`.
   - Transcript segments receive the speaker label with the largest time overlap before transcript
     normalization, so speaker changes can prevent unrelated adjacent segments from being merged.
1. Normalize the transcript for report generation.
   - Adjacent transcript segments are merged only when the time gap is small enough and the speaker
     label allows it.
   - The normalized transcript keeps timestamps and optional speaker labels, but produces cleaner
     paragraph-sized material for sectioning and export.
1. Process video context when the input has video.
   - Scene detection samples the video timeline and writes `scenes.json`.
   - Representative frames are extracted into `frames/`.
   - Transcript segments are aligned to scene ranges so the Markdown, DOCX, and JSON reports can
     associate sections with the right visual frame.
1. Build report sections locally.
   - Video reports are primarily organized by detected scene boundaries.
   - Audio-only reports use transcript timing and gaps to produce a best-effort section structure.
   - This local report is complete even when `--llm` is not used.
1. Optionally polish the report with an LLM.
   - `--llm` runs after the deterministic local report exists.
   - The LLM can polish section transcript text and refine summary bullets, action items, section
     titles, and section TL;DRs.
   - If an LLM step fails, diagnostics and warnings record the fallback and the local report content
     remains available.
1. Write final artifacts and diagnostics.
   - `report.md`, `report.docx`, `report.json`, `transcript.json`, and `diagnostics.json` are
     written at the end of a successful run.
   - Failed runs still try to write `diagnostics.json` once the run directory exists, including the
     failed stage, warnings, timings, and any partial artifacts already produced.
