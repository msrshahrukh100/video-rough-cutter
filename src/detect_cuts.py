"""Step 2: Align marked.txt brackets back to word timestamps → cuts.json.

The text-level marking is done by analyze_cuts.py, which wraps repetitive/
false-start segments in square brackets.  This module parses those brackets
and maps them to (start, end) time ranges using the word timestamps from
transcript.json.
"""

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def normalize(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


def invert_cuts(cut_ranges: list, total_duration: float) -> list:
    """Convert (cut_start, cut_end) pairs into keep-interval dicts."""
    if not cut_ranges:
        return [{"start": 0.0, "end": total_duration}]

    keep = []
    prev_end = 0.0
    for cut_start, cut_end in sorted(cut_ranges):
        if cut_start > prev_end:
            keep.append({"start": prev_end, "end": cut_start})
        prev_end = max(prev_end, cut_end)
    if prev_end < total_duration:
        keep.append({"start": prev_end, "end": total_duration})
    return keep


# ---------------------------------------------------------------------------
# Marked-text parsing
# ---------------------------------------------------------------------------

def parse_marked_text(marked_text: str) -> list[tuple[str, bool]]:
    """Parse text with [...] markers.

    Returns a list of (text, is_cut) tuples — is_cut=True for bracketed
    segments, False for the clean text between them.
    """
    segments = []
    last_end = 0
    for m in re.finditer(r"\[([^\[\]]*)\]", marked_text, re.DOTALL):
        before = marked_text[last_end:m.start()]
        if before:
            segments.append((before, False))
        segments.append((m.group(1), True))
        last_end = m.end()
    tail = marked_text[last_end:]
    if tail:
        segments.append((tail, False))
    return segments


# ---------------------------------------------------------------------------
# Timestamp alignment
# ---------------------------------------------------------------------------

def align_segments_to_timestamps(
    segments: list[tuple[str, bool]],
    words: list[dict],
    padding: float,
) -> list[tuple[float, float]]:
    """Map (text, is_cut) segments onto transcript word timestamps.

    Uses difflib.SequenceMatcher to find the globally optimal alignment
    between flattened marked words and transcript words.  Only "equal"
    (matched) word pairs are used to determine cut boundaries, so
    mismatches in one region cannot cause catastrophic over-cutting
    elsewhere.

    Returns a list of (cut_start, cut_end) pairs with padding applied.
    """
    total_duration = words[-1]["end"] if words else 0.0

    # Flatten segments into (norm_word, is_cut) pairs, skipping empty tokens
    flat: list[tuple[str, bool]] = []
    for text, is_cut in segments:
        for token in text.split():
            norm = normalize(token)
            if norm:
                flat.append((norm, is_cut))

    flat_norms = [f[0] for f in flat]
    trans_norms = [normalize(w["word"]) for w in words]

    # Build flat_index → transcript_index map for every matched word pair.
    # autojunk=False is critical: speech repeats common words ("and", "so",
    # "the") which the default junk heuristic would skip, breaking alignment.
    matcher = SequenceMatcher(None, flat_norms, trans_norms, autojunk=False)
    flat_to_trans: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                flat_to_trans[i] = j

    # Walk cut spans in flat; look up their matched transcript positions
    cut_ranges: list[tuple[float, float]] = []
    in_cut = False
    cut_flat_start = None

    for fi, (_, is_cut) in enumerate(flat):
        if is_cut and not in_cut:
            in_cut = True
            cut_flat_start = fi
        elif not is_cut and in_cut:
            matched = [flat_to_trans[i] for i in range(cut_flat_start, fi)
                       if i in flat_to_trans]
            if matched:
                cs = max(0.0, words[min(matched)]["start"] - padding)
                ce = min(total_duration, words[max(matched)]["end"] + padding)
                cut_ranges.append((cs, ce))
            in_cut = False

    # Flush a trailing cut span
    if in_cut:
        matched = [flat_to_trans[i] for i in range(cut_flat_start, len(flat))
                   if i in flat_to_trans]
        if matched:
            cs = max(0.0, words[min(matched)]["start"] - padding)
            ce = min(total_duration, words[max(matched)]["end"] + padding)
            cut_ranges.append((cs, ce))

    return cut_ranges


# ---------------------------------------------------------------------------
# Entry point used by pipeline.py
# ---------------------------------------------------------------------------

def run(transcript_path: str, output_path: str, total_duration: float,
        padding: float = 0.08, model: str = "gpt-4o",
        marked_path: str = None) -> None:

    data = json.loads(Path(transcript_path).read_text())
    words = data["words"]
    print(f"[detect_cuts] Processing {len(words)} words...")

    transcript_text = " ".join(w["word"] for w in words)

    if marked_path and Path(marked_path).exists():
        marked_text = Path(marked_path).read_text()
        print(f"[detect_cuts] Using existing marked text: {marked_path}")
    else:
        import analyze_cuts
        marked_text = analyze_cuts.mark_repetitions(transcript_text, model)
        if marked_path:
            Path(marked_path).write_text(marked_text)
            print(f"[detect_cuts] Saved marked text to {marked_path}")

    segments = parse_marked_text(marked_text)
    cut_ranges = align_segments_to_timestamps(segments, words, padding)

    for i, (cs, ce) in enumerate(cut_ranges):
        print(f"  cut {i+1}: {cs:.3f}s\u2013{ce:.3f}s")

    keep_intervals = invert_cuts(cut_ranges, total_duration)

    result = {
        "keep_intervals": keep_intervals,
        "cuts_made": len(cut_ranges),
        "total_duration": total_duration,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"[detect_cuts] {len(cut_ranges)} cut(s) → {output_path}")


# ---------------------------------------------------------------------------
# Unit test (no network needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test parse_marked_text
    segments = parse_marked_text("[hi hello] world")
    assert segments == [("hi hello", True), (" world", False)], segments

    # Test align_segments_to_timestamps
    words = [
        {"word": "hi",    "start": 0.0, "end": 0.3},
        {"word": "hello", "start": 0.4, "end": 0.7},
        {"word": "world", "start": 0.9, "end": 1.3},
    ]
    cut_ranges = align_segments_to_timestamps(segments, words, padding=0.0)
    assert len(cut_ranges) == 1, cut_ranges
    assert cut_ranges[0][0] == 0.0
    assert abs(cut_ranges[0][1] - 0.7) < 0.01, cut_ranges

    # Mismatch test: transcript has an extra word in the middle of a cut
    words2 = [
        {"word": "hi",    "start": 0.0,  "end": 0.3},
        {"word": "there", "start": 0.35, "end": 0.5},  # extra word not in marked text
        {"word": "hello", "start": 0.5,  "end": 0.7},
        {"word": "world", "start": 0.9,  "end": 1.3},
    ]
    segments2 = parse_marked_text("[hi hello] world")
    cut_ranges2 = align_segments_to_timestamps(segments2, words2, padding=0.0)
    assert len(cut_ranges2) == 1, cut_ranges2
    assert cut_ranges2[0][0] == 0.0, cut_ranges2          # start of "hi"
    assert abs(cut_ranges2[0][1] - 0.7) < 0.01, cut_ranges2   # end of "hello"

    print("Unit tests passed.")
