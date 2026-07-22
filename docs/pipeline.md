# Pipeline

The CLI builds a deterministic local report through the stages below, then optionally polishes it
with an LLM. For installation and typical usage, see [README.md](../README.md).

1. Create an isolated run directory.
   - Each input gets its own `runs/<timestamp>_<basename>/` directory unless you pass a single
     `--output-dir`.
   - The CLI refuses existing output directories, so reruns cannot silently mix old and new
     artifacts.
1. Probe the media.
   - Every input must contain a decodable audio stream. A usable video stream adds scene processing;
     without one, the input follows the audio-only path.
   - The probe records the media type, duration, and stream metadata available to the rest of the
     pipeline.
   - The pipeline writes this data to `metadata.json` early. Later stages use the probed duration
     for progress, diagnostics, and report timing.
1. Prepare deterministic transcription audio.
   - PyAV decodes the input into temporary mono, 16 kHz, 16-bit [PCM] WAV audio.
   - Downstream speech detection, ASR, and diarization all consume this normalized audio contract.
   - The temporary WAV normally stays outside the run directory. `--keep-audio` saves a compressed
     `transcription-audio.mp3` copy before transcription, so it survives a later-stage failure.
1. Load the local [ASR] runtime.
   - `pywhispercpp` loads the selected [Whisper] (`whisper.cpp`) model identifier or local GGML
     model path.
   - The backend, model name, thread count, and native `whisper.cpp` logs are recorded in the run
     directory so performance and failures can be inspected after the run.
1. Detect speech regions.
   - The bundled Silero [VAD] [ONNX] model runs locally through `sherpa-onnx`.
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
   - The raw per-window result is written to `asr/decoded_windows.json`; use it to debug boundary
     behavior before reconciliation.
1. Reconcile raw windows into a transcript.
   - Overlapping window boundaries are deduplicated by comparing text prefixes and suffixes.
   - Empty or invalid decoded segments are discarded.
   - The reconciled transcript is the source for `transcript.json` and for optional speaker
     assignment.
1. Optionally [diarize][diarization] speakers.
   - `--diarize` runs local `sherpa-onnx` speaker segmentation and embedding models against the same
     normalized audio.
   - Speaker turns are normalized to anonymous labels ordered by first appearance (`S1`, `S2`, and
     so on) and written to `diarization.json`.
   - Transcript segments receive the speaker label with the largest time overlap before transcript
     coalescing, so a speaker change starts a new block.
1. Coalesce the transcript for report generation.
   - Adjacent segments are merged into readable, paragraph-sized blocks; a new block starts on a
     speaker change, a timing gap, or once a block reaches its length or a
     [sentence boundary][sentence-boundary]. Speaker labels only refine the boundaries, so unlabeled
     transcripts are coalesced the same way.
   - The report renders each block as a paragraph and prints a speaker label only when the speaker
     changes, so one speaker turn carries a single label. Coalescing is in-memory; the persisted
     `transcript.json` keeps the raw per-segment transcript.
1. Process video context when the input has video.
   - [Scene detection][scene-detection] samples the video timeline to find slide-change boundaries.
   - A second pass seeks to the middle of each scene and saves a settled frame into `frames/`, so
     the representative image is a full slide rather than a slide-change or fade-in frame;
     `scenes.json` then records each scene's bounds and saved frame.
   - Transcript segments are aligned to scene ranges so the Markdown, DOCX, and JSON reports can
     associate sections with the right visual frame.
1. Build report sections locally.
   - Video reports are primarily organized by detected scene boundaries.
   - Audio-only reports use transcript timing and gaps to produce deterministic sections.
   - The local report is written even when `--llm` is not used.
1. Optionally polish the report with an [LLM].
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

[asr]: https://en.wikipedia.org/wiki/Speech_recognition
[diarization]: https://en.wikipedia.org/wiki/Speaker_diarisation
[llm]: https://en.wikipedia.org/wiki/Large_language_model
[onnx]: https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange
[pcm]: https://en.wikipedia.org/wiki/Pulse-code_modulation
[scene-detection]: https://en.wikipedia.org/wiki/Shot_transition_detection
[sentence-boundary]: https://en.wikipedia.org/wiki/Sentence_boundary_disambiguation
[vad]: https://en.wikipedia.org/wiki/Voice_activity_detection
[whisper]: https://en.wikipedia.org/wiki/Whisper_(speech_recognition_system)
