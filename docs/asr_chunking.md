# ASR Chunking Design

The whisper.cpp path now treats Silero VAD as the primary segmentation planner.

## Flow

1. Silero VAD produces coarse speech regions.
1. Very short neighboring speech regions are merged in Python until they become usable ASR units.
1. The planner expands each region with explicit symmetric ASR padding.
1. Each expanded region becomes one deterministic inference window.
1. whisper.cpp decodes one window at a time and returns structured window results.
1. The Python layer carries a bounded prompt suffix forward only when the previous window looks
   trustworthy.
1. The reconciler stitches adjacent decoded windows into the final transcript.

## Why Plan This Way

- `whispercpp.py` stays thin and policy-free. It only loads the model, decodes one window, and
  returns timestamps and text for a single decode window.
- `segmentation.py` owns planning because short-region repair and padding heuristics are application
  policy, not binding details.
- Silero padding should stay small. The expansion stage is the intended owner of the ASR boundary
  budget, which keeps the merge behavior composable and easier to tune.
- `asr.py` owns carryover because prompt reuse depends on decoding confidence and user-facing tuning
  knobs.
- Long-region subdivision is intentionally avoided. whisper.cpp already segments within each decode
  call, and keeping one planned window per speech region makes the outer planner easier to inspect.

## Carryover Rules

- Carry only the last one or two sentences.
- Sanitize whitespace and trailing partial punctuation.
- Enforce an approximate token budget before sending the prompt to whisper.cpp.
- Drop carryover entirely when the previous window is empty or marked as fallback.

## Default Tuning

The default policy is aimed at offline webinars and remains CLI-configurable:

- VAD threshold stays near the current default.
- Silero padding stays low.
- External ASR padding defaults to `200 ms` per side.
- Very short speech regions are merged until they reach at least `3.0 s`, when neighboring pauses
  stay within the repair gap budget.

## Reconciliation Rules

- Keep transcript assembly in Python rather than in the native binding.
- Rebuild monotonic transcript segments from adjacent decoded windows after ASR finishes.

## Determinism

The planner, carryover builder, and reconciler are all deterministic and local-first. No network
services are involved in the ASR path, and the same input plus configuration should produce the same
planned windows and transcript merge behavior.

## Saved Artifacts

Each successful `process` run now persists the intermediate ASR planning and raw decode stages under
`asr/` in the run directory:

- `speech_regions.json`
- `expanded_regions.json`
- `decoded_windows.json` Each decoded window now includes the `input_prompt` that was sent into that
  decode.
