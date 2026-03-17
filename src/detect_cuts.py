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

def _format_utterances(utterances: list) -> tuple[str, str]:
    """Return (paragraph_view, indexed_view) for the utterance list."""
    paragraph_parts = []
    indexed_lines = []
    for i, utt in enumerate(utterances):
        start = utt[0]["start"]
        end = utt[-1]["end"]
        text = " ".join(w["word"] for w in utt)
        paragraph_parts.append(f"[{i}]{text}")
        indexed_lines.append(f'{i}: [{start:.2f}s\u2013{end:.2f}s] "{text}"')
    paragraph_view = " ".join(paragraph_parts)
    indexed_view = "\n".join(indexed_lines)
    return paragraph_view, indexed_view


def _infer_provider(model: str) -> str:
    """Infer the LLM provider from the model name."""
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "gemini"
    raise ValueError(
        f"Cannot infer provider from model name '{model}'. "
        "Use a model name starting with 'gpt-'/'o1'/'o3' (OpenAI), "
        "'claude-' (Anthropic), or 'gemini-' (Google)."
    )


def _parse_json(text: str) -> dict:
    """Parse JSON that may be wrapped in markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text.strip())
    return json.loads(text)


def _call_llm(system_prompt: str, user_prompt: str, model: str) -> dict:
    """Send system+user prompt to the appropriate LLM provider and return parsed JSON."""
    provider = _infer_provider(model)

    if provider == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Set it or use --no-ai.")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)

    if provider == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Set it or use --no-ai.")
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0,
        )
        return _parse_json(response.content[0].text)

    if provider == "gemini":
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set. Set it or use --no-ai.")
        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
            system_instruction=system_prompt,
        )
        return json.loads(gen_model.generate_content(user_prompt).text)


def detect_cuts_ai(words: list, total_duration: float, padding: float = 0.08,
                   model: str = "gpt-4o") -> list:
    """
    Group words into utterances, send to an LLM, return (cut_start, cut_end) pairs.

    The AI works by reading the full stitched transcript as a paragraph, identifying
    any text that should not be in the final output (repetitions, restarts, fillers,
    incoherent fragments), then returning the utterance indices to remove.

    Supported providers (inferred from model name):
      - OpenAI:    gpt-*, o1*, o3*  →  OPENAI_API_KEY
      - Anthropic: claude-*         →  ANTHROPIC_API_KEY
      - Gemini:    gemini-*         →  GEMINI_API_KEY
    """
    utterances = group_into_utterances(words)

    if len(utterances) <= 1:
        print("[detect_cuts] Only one utterance — no cuts possible.")
        return []

    paragraph_view, indexed_view = _format_utterances(utterances)

    system_prompt = f"""\
You are a transcript editor for video content. Your task: given a raw speech transcript, \
produce a clean version that reads like polished, natural English — no repetitions, no \
broken fragments, no filler noise.

════════════════════════════════════════════════════════
STEP 1 — READ THE TRANSCRIPT AS A PARAGRAPH
════════════════════════════════════════════════════════
You will be given the transcript in two forms:

  PARAGRAPH VIEW: all utterances concatenated in order, each prefixed with [index].
  Read this as if it were a written document. Notice every place where the text \
repeats itself, restarts, or doesn't make sense.

  INDEXED VIEW: each utterance with its timestamp, for reference when building your answer.

════════════════════════════════════════════════════════
STEP 2 — FIND EVERYTHING THAT SHOULD NOT BE IN THE FINAL TEXT
════════════════════════════════════════════════════════
As you read the paragraph, mark any utterance that falls into one of these categories:

  REPETITION / RESTART
  The same phrase or sentence appears more than once in a row because the speaker
  restarted. Keep only the LAST (most complete) attempt; mark all earlier ones.
    "[0]hi and welcome [1]hi and welcome to [2]hi and welcome to a new lesson"
    → mark [0] and [1] for removal, keep [2]

  STUTTER / IMMEDIATE REPEAT
  A word or short phrase is repeated right before the speaker continues.
    "[5]let's start [6]let's start building the app"
    → mark [5] for removal, keep [6]

  ABANDONED FRAGMENT
  The speaker starts a sentence, stops, and then says something different.
  The abandoned start adds nothing and makes the text confusing.
    "[8]so the way that [9]actually let me show you a different approach"
    → mark [8] for removal (it was abandoned and replaced by [9])

  PURE FILLER
  An utterance whose entire content is noise: "um", "uh", "er", "like", \
"you know", "okay so", "and um", etc. with zero real information.
    "[3]um [7]so like you know"  →  mark [3] and [7] for removal
  DO NOT mark utterances that start with a filler but then say something real.

  INCOHERENT FRAGMENT
  An utterance that makes no grammatical or semantic sense in context — garbled \
words, contradictory noise, or a partial thought that was never completed and \
never restarted.
    "[12]yeah no so the"  →  mark [12] for removal

════════════════════════════════════════════════════════
STEP 3 — VALIDATE BY READING THE CLEANED PARAGRAPH
════════════════════════════════════════════════════════
Take the paragraph view, mentally delete every utterance you have marked, and \
re-read the result. It must satisfy ALL of these:

  ✓ No phrase or sentence appears more than once in a row
  ✓ Every sentence is grammatically complete and makes sense
  ✓ The text flows naturally from one sentence to the next
  ✓ No broken or abandoned fragments remain

If anything still fails, add more indices to your removal list before answering.

════════════════════════════════════════════════════════
STEP 4 — TRIM LONG SILENCES
════════════════════════════════════════════════════════
After deciding removals, also check for gaps between consecutive utterances that \
exceed ~2 seconds. Those feel like dead air. Add the index of the utterance \
BEFORE the gap to "trim_silence_after". The gap will be reduced to {_KEEP_SILENCE}s \
automatically. Do NOT flag natural pauses of 1–2s.

════════════════════════════════════════════════════════
STRICT RULES
════════════════════════════════════════════════════════
- When a phrase is restarted multiple times, keep only the LAST version.
- Do NOT remove an utterance just because it sounds informal or has a filler word \
at the start — only remove it if the cleaned paragraph is genuinely better without it.
- Do NOT remove deliberate repetition for emphasis or a recap of a previous point.
- If you are not confident an utterance should be removed, leave it in.

════════════════════════════════════════════════════════
RESPONSE FORMAT
════════════════════════════════════════════════════════
Return ONLY valid JSON with exactly these two keys (use [] if nothing to report):

{{
  "remove": [<int>, ...],
  "trim_silence_after": [<int>, ...]
}}

"remove"             → indices of all utterances to delete (repetitions, fillers, \
fragments, incoherent noise — everything from Steps 2–3)
"trim_silence_after" → indices of utterances followed by a gap that is too long (Step 4)
"""

    user_prompt = (
        f"PARAGRAPH VIEW ({len(utterances)} utterances):\n\n"
        f"{paragraph_view}\n\n"
        f"INDEXED VIEW (with timestamps):\n\n"
        f"{indexed_view}\n\n"
        "Follow Steps 1–4 and return the JSON."
    )

    provider = _infer_provider(model)
    print(f"[detect_cuts] Sending {len(utterances)} utterances to {provider} ({model})...")

    result = _call_llm(system_prompt, user_prompt, model)
    n = len(utterances)
    cut_ranges = []

    # --- Removals (repetitions, fillers, fragments, incoherent noise) ---
    for idx in result.get("remove", []):
        if not (0 <= idx < n):
            print(f"  [warn] Skipping invalid remove index: {idx}")
            continue
        cs = max(0.0, utterances[idx][0]["start"] - padding)
        ce = utterances[idx][-1]["end"] + padding
        cut_ranges.append((cs, ce))
        text = " ".join(w["word"] for w in utterances[idx])
        print(f"  [remove]  {cs:.2f}s\u2013{ce:.2f}s  \"{text[:70]}\"")

    # --- Long silences ---
    for idx in result.get("trim_silence_after", []):
        if not (0 <= idx < n - 1):
            print(f"  [warn] Skipping invalid trim_silence_after index: {idx}")
            continue
        gap_start = utterances[idx][-1]["end"]
        gap_end   = utterances[idx + 1][0]["start"]
        cs = gap_start + 0.05
        ce = gap_end - _KEEP_SILENCE
        if ce > cs:
            cut_ranges.append((cs, ce))
            print(f"  [silence] {cs:.2f}s\u2013{ce:.2f}s  (gap after utterance {idx}: {gap_end - gap_start:.1f}s → {_KEEP_SILENCE}s)")

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
        model: str = "gpt-4o", min_match: int = 2) -> None:

    data = json.loads(Path(transcript_path).read_text())
    words = data["words"]
    print(f"[detect_cuts] Processing {len(words)} words...")

    if use_ai:
        cut_ranges = detect_cuts_ai(
            words, total_duration, padding=padding, model=model
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
