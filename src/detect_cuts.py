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

    Aligns only KEEP segments sequentially against the transcript (cursor-based,
    left-to-right), then derives cut ranges as the gaps between consecutive keep
    extents.  Cut segment text is never aligned directly — this avoids the
    global-alignment failure where cut words (which are often verbatim
    repetitions of adjacent keep text) get matched to the wrong transcript
    occurrence.

    Returns a list of (cut_start, cut_end) pairs with padding applied.
    """
    total_duration = words[-1]["end"] if words else 0.0
    if not segments or not words:
        return []

    trans_norms = [normalize(w["word"]) for w in words]

    # Build ordered keep-item list: (norm_words, has_cut_before).
    # has_cut_before is True when at least one CUT segment appears between
    # the previous KEEP and this one (or before the very first KEEP).
    keep_items: list[tuple[list[str], bool]] = []
    pending_cut = segments[0][1]  # True if the sequence opens with a cut
    for text, is_cut in segments:
        if not is_cut:
            seg_norms = [normalize(t) for t in text.split() if normalize(t)]
            if seg_norms:
                keep_items.append((seg_norms, pending_cut))
            pending_cut = False
        else:
            pending_cut = True
    trailing_cut = segments[-1][1]

    if not keep_items:
        # Entire transcript is marked for cutting
        return [(0.0, total_duration)]

    # Step 1: align each keep segment to transcript[cursor:], left-to-right.
    # autojunk=False is critical: common filler words ("and", "so", "the")
    # must not be treated as junk or the alignment breaks on speech repetitions.
    keep_extents: list[tuple[int, int, bool]] = []  # (abs_start, abs_end, has_cut_before)
    cursor = 0
    for seg_norms, has_cut_before in keep_items:
        matcher = SequenceMatcher(None, seg_norms, trans_norms[cursor:], autojunk=False)
        matched_j: list[int] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                matched_j.extend(range(j1, j2))
        if matched_j:
            abs_start = cursor + min(matched_j)
            abs_end = cursor + max(matched_j)
            keep_extents.append((abs_start, abs_end, has_cut_before))
            cursor = abs_end + 1
        # If a segment cannot be matched, leave cursor unchanged so the next
        # segment still searches from the same position.

    if not keep_extents:
        return [(0.0, total_duration)]

    # Step 2: build cut ranges from gaps between keep extents.
    cut_ranges: list[tuple[float, float]] = []

    # Leading cut: before the first keep extent
    if keep_extents[0][2] and keep_extents[0][0] > 0:
        ce = words[keep_extents[0][0]]["start"]
        if ce > 0.0:
            cut_ranges.append((0.0, ce))

    # Gaps between consecutive keep extents (only where there is a cut between them)
    for i in range(len(keep_extents) - 1):
        k1_end = keep_extents[i][1]
        k2_start, _, k2_has_cut = keep_extents[i + 1]
        if k2_has_cut and k2_start > k1_end + 1:
            cs = max(0.0, words[k1_end + 1]["start"] - padding)
            ce = words[k2_start]["start"]
            if cs < ce:
                cut_ranges.append((cs, ce))

    # Trailing cut: after the last keep extent
    if trailing_cut and keep_extents[-1][1] < len(words) - 1:
        trail_idx = keep_extents[-1][1] + 1
        cs = max(0.0, words[trail_idx]["start"] - padding)
        if cs < total_duration:
            cut_ranges.append((cs, total_duration))

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

    # Basic leading cut: [cut] keep
    # Cut covers from 0.0 to the start of "world".
    words = [
        {"word": "hi",    "start": 0.0, "end": 0.3},
        {"word": "hello", "start": 0.4, "end": 0.7},
        {"word": "world", "start": 0.9, "end": 1.3},
    ]
    cut_ranges = align_segments_to_timestamps(segments, words, padding=0.0)
    assert len(cut_ranges) == 1, cut_ranges
    assert cut_ranges[0][0] == 0.0, cut_ranges
    assert abs(cut_ranges[0][1] - 0.9) < 0.01, cut_ranges   # start of "world"

    # Middle cut: keep [cut] keep
    # Cut covers from the first transcript word after "intro" to the start of "world".
    words_mid = [
        {"word": "intro", "start": 0.0, "end": 0.2},
        {"word": "hi",    "start": 0.3, "end": 0.6},
        {"word": "hello", "start": 0.7, "end": 1.0},
        {"word": "world", "start": 1.2, "end": 1.6},
    ]
    segs_mid = parse_marked_text("intro [hi hello] world")
    cut_ranges_mid = align_segments_to_timestamps(segs_mid, words_mid, padding=0.0)
    assert len(cut_ranges_mid) == 1, cut_ranges_mid
    assert abs(cut_ranges_mid[0][0] - 0.3) < 0.01, cut_ranges_mid  # start of "hi"
    assert abs(cut_ranges_mid[0][1] - 1.2) < 0.01, cut_ranges_mid  # start of "world"

    # Gap words after cut: transcript has extra words between cut and keep.
    # Those gap words fall in the keep→keep gap and must be included in the cut.
    words_gap_after = [
        {"word": "hi",    "start": 0.0,  "end": 0.3},
        {"word": "gap1",  "start": 0.4,  "end": 0.5},  # omitted from marked.txt
        {"word": "gap2",  "start": 0.55, "end": 0.7},  # omitted from marked.txt
        {"word": "world", "start": 0.9,  "end": 1.3},
    ]
    segs_gap_after = parse_marked_text("[hi] world")
    cut_ranges_gap_after = align_segments_to_timestamps(segs_gap_after, words_gap_after, padding=0.0)
    assert len(cut_ranges_gap_after) == 1, cut_ranges_gap_after
    assert cut_ranges_gap_after[0][0] == 0.0, cut_ranges_gap_after   # leading cut starts at 0
    assert abs(cut_ranges_gap_after[0][1] - 0.9) < 0.01, cut_ranges_gap_after  # start of "world"

    # Gap words before cut: transcript has extra words between keep and cut.
    # Those gap words also fall in the keep→keep gap and must be included in the cut.
    words_gap_before = [
        {"word": "intro", "start": 0.0,  "end": 0.2},
        {"word": "gap1",  "start": 0.3,  "end": 0.4},  # omitted from marked.txt
        {"word": "hi",    "start": 0.5,  "end": 0.8},
        {"word": "world", "start": 1.0,  "end": 1.3},
    ]
    segs_gap_before = parse_marked_text("intro [hi] world")
    cut_ranges_gap_before = align_segments_to_timestamps(segs_gap_before, words_gap_before, padding=0.0)
    assert len(cut_ranges_gap_before) == 1, cut_ranges_gap_before
    assert abs(cut_ranges_gap_before[0][0] - 0.3) < 0.01, cut_ranges_gap_before  # start of "gap1"
    assert abs(cut_ranges_gap_before[0][1] - 1.0) < 0.01, cut_ranges_gap_before  # start of "world"

    # Repeated-phrase cut: the key regression case.
    # The cut text is identical to the keep text immediately before it.
    # The new sequential algorithm must assign the cut to the SECOND occurrence.
    words_repeat = [
        {"word": "so",     "start": 0.0,  "end": 0.2},
        {"word": "hello",  "start": 0.3,  "end": 0.6},  # first occurrence (keep)
        {"word": "so",     "start": 0.7,  "end": 0.9},
        {"word": "hello",  "start": 1.0,  "end": 1.3},  # second occurrence (cut)
        {"word": "world",  "start": 1.5,  "end": 1.8},
    ]
    segs_repeat = parse_marked_text("so hello [so hello] world")
    cut_ranges_repeat = align_segments_to_timestamps(segs_repeat, words_repeat, padding=0.0)
    assert len(cut_ranges_repeat) == 1, cut_ranges_repeat
    assert abs(cut_ranges_repeat[0][0] - 0.7) < 0.01, cut_ranges_repeat  # start of second "so"
    assert abs(cut_ranges_repeat[0][1] - 1.5) < 0.01, cut_ranges_repeat  # start of "world"

    # Trailing cut: keep [cut] — cut extends to total_duration.
    words_trail = [
        {"word": "intro", "start": 0.0, "end": 0.4},
        {"word": "bye",   "start": 0.6, "end": 0.9},
        {"word": "now",   "start": 1.0, "end": 1.4},
    ]
    segs_trail = parse_marked_text("intro [bye now]")
    cut_ranges_trail = align_segments_to_timestamps(segs_trail, words_trail, padding=0.0)
    assert len(cut_ranges_trail) == 1, cut_ranges_trail
    assert abs(cut_ranges_trail[0][0] - 0.6) < 0.01, cut_ranges_trail  # start of "bye"
    assert abs(cut_ranges_trail[0][1] - 1.4) < 0.01, cut_ranges_trail  # total_duration

    print("Unit tests passed.")
