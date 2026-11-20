"""Generate a two-slide, two-voice webinar and score a local CLI run."""

import argparse
import json
import re
import shutil
import subprocess
import wave
from difflib import SequenceMatcher
from pathlib import Path

OPENING = (
    "Today we will measure the cost of a memory request. "
    "The processor asks for one cache line and waits for the data to arrive. "
    "A nearby cache can answer in a few cycles. A trip to main memory takes longer. "
    "We measure latency by following a chain of dependent loads. Each address comes from the "
    "previous result. This prevents the processor from hiding the delay with independent work. "
    "We start with a small array, repeat the experiment with a larger array, and record the "
    "median time for each size. The sharp increase tells us where the working set outgrows "
    "the cache."
)
CLOSING = (
    "Now look at the second slide. Bandwidth answers a different question. "
    "Several independent requests can make progress at the same time. "
    "Count the total bytes transferred and divide by elapsed time."
)


def ffmpeg(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *args], check=True, cwd=cwd)  # noqa: S603,S607


def generate(directory: Path, font_file: Path | None = None) -> int:
    font_option = ""
    if font_file is not None:
        shutil.copyfile(font_file, directory / "slide-font.ttf")
        font_option = "fontfile=slide-font.ttf:"
    for name, voice, script in [("opening", "slt", OPENING), ("closing", "rms", CLOSING)]:
        text = directory / f"{name}.txt"
        text.write_text(script, encoding="utf-8")
        # Use a relative filename in the filter expression so special characters in the output
        # directory never become filter syntax
        ffmpeg(
            "-f",
            "lavfi",
            "-i",
            f"flite=textfile={text.name}:voice={voice}",
            "-ar",
            "16000",
            f"{name}.wav",
            cwd=directory,
        )
    with wave.open(str(directory / "opening.wav")) as audio:
        change = 2 + 2 * ((audio.getnframes() // audio.getframerate()) // 2 + 1)
    with wave.open(str(directory / "closing.wav")) as audio:
        remaining = 2 + audio.getnframes() // audio.getframerate()
    slides = []
    for label, color, duration in [
        ("Memory latency", "0x172C42", change),
        ("Memory bandwidth", "0xF2EBDF", remaining),
    ]:
        ink = "white" if label == "Memory latency" else "black"
        slides.extend([
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=640x360:r=5:d={duration},drawtext={font_option}text='{label}':fontsize=36:fontcolor={ink}:x=35:y=150",
        ])
    ffmpeg(
        *slides,
        "-i",
        "opening.wav",
        "-i",
        "closing.wav",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[v];[2:a]adelay=2000[a];"
        f"[3:a]adelay={change * 1000}[b];[a][b]amix=inputs=2:normalize=0[audio]",
        "-map",
        "[v]",
        "-map",
        "[audio]",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "webinar.mp4",
        cwd=directory,
    )
    return change


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New benchmark directory")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cli", default="webinar-transcriber")
    parser.add_argument("--font-file", type=Path, help="TrueType font for slide titles")
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True)
    change = generate(directory, args.font_file)
    run = directory / "run"
    subprocess.run(  # noqa: S603 - run the selected local CLI
        [
            args.cli,
            str(directory / "webinar.mp4"),
            "--asr-model",
            args.model,
            "--language",
            "en",
            "--diarize",
            "--output-dir",
            str(run),
        ],
        check=True,
    )
    segments = json.loads((run / "transcript.json").read_text())["segments"]

    def words(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    reference = words(OPENING + " " + CLOSING)
    observed = words(" ".join(s["text"] for s in segments))
    matches = SequenceMatcher(None, reference, observed, autojunk=False).get_matching_blocks()
    matched = sum(block.size for block in matches)
    scenes = json.loads((run / "scenes.json").read_text())
    diagnostics = json.loads((run / "diagnostics.json").read_text())
    score = {
        "reference_words": len(reference),
        "unmatched_reference_words": len(reference) - matched,
        "unmatched_output_words": len(observed) - matched,
        "slide_count": len(scenes),
        "expected_slide_count": 2,
        "second_slide_error_sec": scenes[1]["start_sec"] - change if len(scenes) > 1 else None,
        "expected_speaker_count": 2,
        "speaker_count": diagnostics["diarization"]["speaker_count"],
        "asr_windows": diagnostics["item_counts"]["windows"],
        "stage_durations_sec": diagnostics["stage_durations_sec"],
    }
    (directory / "scores.json").write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(score, indent=2))


if __name__ == "__main__":
    main()
