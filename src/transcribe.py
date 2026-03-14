"""Step 1: Transcribe video with Whisper, producing word-level timestamps."""

import json
import sys
from pathlib import Path


def transcribe(video_path: str, model_name: str, output_path: str) -> None:
    import whisper

    print(f"[transcribe] Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    print(f"[transcribe] Transcribing {video_path}...")
    result = model.transcribe(video_path, word_timestamps=True, verbose=False)

    words = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append({
                "word": w["word"].strip().lower(),
                "start": w["start"],
                "end": w["end"],
            })

    data = {"words": words, "text": result["text"]}
    Path(output_path).write_text(json.dumps(data, indent=2))
    print(f"[transcribe] Saved {len(words)} words → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: transcribe.py <video_path> <model_name> <output_path>")
        sys.exit(1)
    transcribe(sys.argv[1], sys.argv[2], sys.argv[3])
