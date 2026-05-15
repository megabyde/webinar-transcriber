# Third-Party Notices

This project vendors the Silero VAD ONNX model at `webinar_transcriber/assets/silero_vad.onnx` and a
trimmed speech fixture at `webinar_transcriber/tests/fixtures/speech-sample.wav`.

- Source: <https://github.com/snakers4/silero-vad>
- License: MIT
- Use: local speech-region detection through `sherpa-onnx`; regression testing for the real VAD
  model

The model is included so the default local VAD path does not require a first-run network download.

The project also depends on these major runtime components:

- PyAV
  - Source: <https://github.com/PyAV-Org/PyAV>
  - License: BSD-3-Clause for PyAV; FFmpeg libraries used by PyAV are licensed separately, commonly
    under LGPL/GPL terms depending on the linked build.
  - Use: deterministic media probing, audio preparation, and video frame extraction.
- sherpa-onnx
  - Source: <https://github.com/k2-fsa/sherpa-onnx>
  - License: Apache-2.0
  - Use: local Silero VAD runtime and optional local speaker diarization.
- whisper.cpp
  - Source: <https://github.com/ggml-org/whisper.cpp>
  - License: MIT
  - Use: local Whisper model inference through `pywhispercpp`.
