"""Reconstruct transcript.json as readable paragraph text."""

import json
import sys
from pathlib import Path


def transcript_to_text(transcript_path: str) -> str:
    data = json.loads(Path(transcript_path).read_text())

    # Prefer the top-level text field if present
    if data.get("text", "").strip():
        return data["text"].strip()

    # Fall back to joining words
    words = [w["word"] for w in data.get("words", [])]
    return " ".join(words).strip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: transcript_to_text.py <transcript.json>")
        sys.exit(1)
    print(transcript_to_text(sys.argv[1]))
