# Third-Party Notices

This project vendors the Silero VAD ONNX model at
`webinar_transcriber/assets/silero_vad.onnx` and a trimmed speech fixture at
`webinar_transcriber/tests/fixtures/speech-sample.wav`.

- Source: <https://github.com/snakers4/silero-vad>
- License: MIT
- Use: local speech-region detection through `sherpa-onnx`; regression testing for the real VAD
  model

The model is included so the default local VAD path does not require a first-run network download.
