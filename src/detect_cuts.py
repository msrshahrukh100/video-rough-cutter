"""Step 2: Detect repetitions/false-starts and produce keep intervals.

Default mode: uses OpenAI to semantically identify what to cut.
Fallback mode (--no-ai): uses greedy word-match algorithm (exact matches only).
"""

import json
import os
import re
from pathlib import Path

# Silence gap threshold for grouping words into utterances
_SILENCE_GAP = 0.4  # seconds

# How much silence to preserve after trimming a long pause
_KEEP_SILENCE = 0.4  # seconds


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def normalize(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


def group_into_utterances(words: list, gap: float = _SILENCE_GAP) -> list[list]:
    """Split word list into utterances wherever there is a silence >= gap seconds."""
    if not words:
        return []
    utterances = []
    current = [words[0]]
    for word in words[1:]:
        if word["start"] - current[-1]["end"] > gap:
            utterances.append(current)
            current = []
        current.append(word)
    if current:
        utterances.append(current)
    return utterances


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
# AI-based detection (default)
# ---------------------------------------------------------------------------

def _format_utterances(utterances: list) -> str:
    lines = []
    for i, utt in enumerate(utterances):
        start = utt[0]["start"]
        end = utt[-1]["end"]
        text = " ".join(w["word"] for w in utt)
        # Show the gap to the NEXT utterance so the AI can judge silence length
        lines.append(f'{i}: [{start:.2f}s\u2013{end:.2f}s] "{text}"')
    return "\n".join(lines)


def detect_cuts_ai(words: list, total_duration: float, padding: float = 0.08,
                   openai_model: str = "gpt-4o") -> list:
    """
    Group words into utterances, send to OpenAI, return (cut_start, cut_end) pairs.

    The AI identifies four things to remove:
      1. Repetition chains (progressive restarts)
      2. Mid-sentence stutters
      3. Semantic corrections (same idea restated)
      4. Filler-only utterances and long silences between utterances
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it or use --no-ai to fall back to the algorithm."
        )

    client = OpenAI(api_key=api_key)
    utterances = group_into_utterances(words)

    if len(utterances) <= 1:
        print("[detect_cuts] Only one utterance — no cuts possible.")
        return []

    transcript_text = _format_utterances(utterances)

    system_prompt = f"""\
You are a strict video editing assistant. Given a numbered list of speech utterances with \
timestamps, you identify everything that should be cut so the final video is clean and \
tight. You handle four types of removal:

─────────────────────────────────────────────────────────
TYPE 1 · PROGRESSIVE RESTART
─────────────────────────────────────────────────────────
The speaker re-says a phrase from its start, each time extending it further.
  0: [0.10s–0.30s] "hi and welcome"
  1: [0.80s–1.20s] "hi and welcome to"
  2: [1.50s–3.20s] "hi and welcome to a new lesson in this video"
→ cuts: first=0, last_kept=2  (remove 0–1, keep 2)

─────────────────────────────────────────────────────────
TYPE 2 · MID-SENTENCE STUTTER / IMMEDIATE REPEAT
─────────────────────────────────────────────────────────
The speaker repeats a fragment right where they are before continuing.
  0: [5.00s–5.40s] "let's start"
  1: [5.50s–7.80s] "let's start building the app"
→ cuts: first=0, last_kept=1

─────────────────────────────────────────────────────────
TYPE 3 · SEMANTIC CORRECTION
─────────────────────────────────────────────────────────
The speaker restates the same instruction with different or more precise words. \
The second version supersedes the first.
  0: [12.00s–12.60s] "let's open our IDE"
  1: [13.00s–14.20s] "let's open our AI coding IDE"
→ cuts: first=0, last_kept=1

  DO cut:     "let's open our IDE" → "let's open our AI coding IDE"
  DO NOT cut: "let's open our IDE" … [other content] … "our AI coding IDE is great" \
(second is a new sentence, not a restart)

─────────────────────────────────────────────────────────
TYPE 4 · FILLER-ONLY UTTERANCES
─────────────────────────────────────────────────────────
An utterance whose entire content is filler sounds or words with no real information: \
"um", "uh", "er", "like", "so", "right", "you know", "okay so", "and um", etc.
Remove the whole utterance.
  3: [18.00s–18.30s] "um"
  7: [34.10s–34.60s] "so like you know"
→ remove: [3, 7]

  DO remove:  utterances that are purely filler with zero content
  DO NOT remove: utterances that START with a filler but then say something real \
("um, so the next step is…" — keep it)

─────────────────────────────────────────────────────────
TYPE 5 · LONG SILENCE BETWEEN UTTERANCES
─────────────────────────────────────────────────────────
If the gap between the END of one utterance and the START of the next exceeds ~2 seconds, \
trim that gap. Signal this by including the earlier utterance's index in "trim_silence_after". \
The gap will be reduced to {_KEEP_SILENCE}s automatically; you do not specify the exact amount to cut.
  4: [22.00s–22.80s] "and here is the result"
  5: [27.50s–29.00s] "now let's move on"   ← 4.7s gap after utterance 4
→ trim_silence_after: [4]

  Only flag gaps that feel awkward or dead — roughly 2s or more.
  Do NOT flag natural pauses of 1–2s that give the viewer breathing room.

─────────────────────────────────────────────────────────
GLOBAL STRICT RULES — when in doubt, do NOT cut
─────────────────────────────────────────────────────────
- Repetitions (types 1–3) must be CONSECUTIVE with no unrelated content between them.
- When a chain has 3+ attempts, keep only the LAST one.
- The goal is that after all cuts are applied, the remaining utterances stitch together \
into complete, natural sentences — as if the speaker said it perfectly the first time. \
Every cut must leave the kept utterances grammatically and semantically continuous with \
what immediately precedes and follows them.
- Do NOT cut natural continuations, rhetorical repetition for emphasis, or deliberate \
re-caps of a previous point.
- Do NOT remove utterances that contain any real content, even if they also have fillers.
- If you are not confident, leave it alone.

─────────────────────────────────────────────────────────
RESPONSE FORMAT
─────────────────────────────────────────────────────────
Return ONLY a JSON object with these three keys (all required, use [] if nothing to report):

{{
  "cuts": [
    {{"first": <int>, "last_kept": <int>}}
  ],
  "remove": [<int>, ...],
  "trim_silence_after": [<int>, ...]
}}

"cuts"               → restart/correction chains; first=index of first bad utterance, \
last_kept=index of final keeper (last_kept > first)
"remove"             → indices of filler-only utterances to delete entirely
"trim_silence_after" → indices of utterances followed by a gap that is too long
"""

    user_prompt = (
        f"Transcript utterances ({len(utterances)} total):\n\n"
        f"{transcript_text}\n\n"
        "Analyze and return the JSON."
    )

    print(f"[detect_cuts] Sending {len(utterances)} utterances to OpenAI ({openai_model})...")

    response = client.chat.completions.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    result = json.loads(response.choices[0].message.content)
    n = len(utterances)
    cut_ranges = []

    # --- TYPE 1–3: repetition/correction chains ---
    for cut in result.get("cuts", []):
        first, last_kept = cut["first"], cut["last_kept"]
        if not (0 <= first < last_kept < n):
            print(f"  [warn] Skipping invalid cut: first={first}, last_kept={last_kept}")
            continue
        cs = max(0.0, utterances[first][0]["start"] - padding)
        ce = max(cs, utterances[last_kept][0]["start"] - padding)
        cut_ranges.append((cs, ce))
        first_text = " ".join(w["word"] for w in utterances[first])
        kept_text  = " ".join(w["word"] for w in utterances[last_kept])
        print(f"  [repetition] {cs:.2f}s\u2013{ce:.2f}s  removes: \"{first_text[:45]}\"  keeps: \"{kept_text[:45]}\"")

    # --- TYPE 4: filler-only utterances ---
    for idx in result.get("remove", []):
        if not (0 <= idx < n):
            print(f"  [warn] Skipping invalid remove index: {idx}")
            continue
        cs = max(0.0, utterances[idx][0]["start"] - padding)
        ce = utterances[idx][-1]["end"] + padding
        cut_ranges.append((cs, ce))
        text = " ".join(w["word"] for w in utterances[idx])
        print(f"  [filler]     {cs:.2f}s\u2013{ce:.2f}s  removes: \"{text[:60]}\"")

    # --- TYPE 5: long silences ---
    for idx in result.get("trim_silence_after", []):
        if not (0 <= idx < n - 1):
            print(f"  [warn] Skipping invalid trim_silence_after index: {idx}")
            continue
        gap_start = utterances[idx][-1]["end"]
        gap_end   = utterances[idx + 1][0]["start"]
        cs = gap_start + 0.05          # tiny buffer after last word
        ce = gap_end - _KEEP_SILENCE   # leave _KEEP_SILENCE seconds before next word
        if ce > cs:
            cut_ranges.append((cs, ce))
            print(f"  [silence]    {cs:.2f}s\u2013{ce:.2f}s  (gap after utterance {idx}: {gap_end - gap_start:.1f}s → {_KEEP_SILENCE}s)")

    print(f"[detect_cuts] Total cuts: {len(cut_ranges)}")
    return cut_ranges


# ---------------------------------------------------------------------------
# Algorithm-based detection (fallback, --no-ai)
# ---------------------------------------------------------------------------

def detect_cuts_algorithm(words: list, min_match: int = 2, padding: float = 0.08) -> list:
    """
    Greedy forward scan: at each position i, find the LATEST position j > i
    where words[i:i+k] exactly matches words[j:j+k] for k >= min_match.

    Limitation: exact word matches only — use AI mode for real-world speech.
    """
    n = len(words)
    texts = [normalize(w["word"]) for w in words]
    cut_ranges = []

    i = 0
    while i < n:
        best_j = -1
        for j in range(i + 1, n):
            k = 0
            while (i + k < n and j + k < n
                   and texts[i + k] == texts[j + k]
                   and texts[i + k] != ""):
                k += 1
            if k >= min_match:
                best_j = j
        if best_j != -1:
            cs = max(0.0, words[i]["start"] - padding)
            ce = max(cs, words[best_j]["start"] - padding)
            cut_ranges.append((cs, ce))
            i = best_j
        else:
            i += 1

    return cut_ranges


# ---------------------------------------------------------------------------
# Entry point used by pipeline.py
# ---------------------------------------------------------------------------

def run(transcript_path: str, output_path: str, total_duration: float,
        padding: float = 0.08, use_ai: bool = True,
        openai_model: str = "gpt-4o", min_match: int = 2) -> None:

    data = json.loads(Path(transcript_path).read_text())
    words = data["words"]
    print(f"[detect_cuts] Processing {len(words)} words...")

    if use_ai:
        cut_ranges = detect_cuts_ai(
            words, total_duration, padding=padding, openai_model=openai_model
        )
    else:
        print("[detect_cuts] Using algorithmic mode (--no-ai).")
        cut_ranges = detect_cuts_algorithm(words, min_match=min_match, padding=padding)
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
# Unit test (algorithm mode, no network needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    words = [
        {"word": "my",       "start": 0.1, "end": 0.3},
        {"word": "name",     "start": 0.3, "end": 0.6},
        {"word": "my",       "start": 0.8, "end": 1.0},
        {"word": "name",     "start": 1.0, "end": 1.3},
        {"word": "is",       "start": 1.3, "end": 1.5},
        {"word": "shahrukh", "start": 1.5, "end": 2.0},
    ]
    cuts = detect_cuts_algorithm(words, min_match=2)
    print("Algorithm unit test cuts:", cuts)
    assert len(cuts) == 1
    assert abs(cuts[0][0] - 0.02) < 0.01
    assert abs(cuts[0][1] - 0.72) < 0.01
    print("Unit test passed.")
