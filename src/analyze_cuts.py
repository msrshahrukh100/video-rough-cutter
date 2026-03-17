"""Analyze a plain-text transcript and mark repetitive / false-start segments.

The marked segments are wrapped in square brackets [ ].  Removing every
bracketed segment leaves behind coherent, well-structured English.

Usage
-----
  # Analyze a text file
  python analyze_cuts.py transcript.txt

  # Pipe text in
  echo "hello hello world" | python analyze_cuts.py

  # Choose a specific model (same --ai-model flag as cut.sh)
  python analyze_cuts.py transcript.txt --ai-model claude-sonnet-4-6
  python analyze_cuts.py transcript.txt --ai-model gemini-1.5-pro
"""

import argparse
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# LLM helpers  (mirrors detect_cuts.py — same provider-inference logic)
# ---------------------------------------------------------------------------

def _infer_provider(model: str) -> str:
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "gemini"
    raise ValueError(
        f"Cannot infer provider from model name '{model}'. "
        "Use a name starting with 'gpt-'/'o1'/'o3' (OpenAI), "
        "'claude-' (Anthropic), or 'gemini-' (Google)."
    )


def _call_llm(system_prompt: str, user_prompt: str, model: str) -> str:
    """Send prompts to the appropriate provider and return the raw text reply."""
    provider = _infer_provider(model)

    if provider == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    if provider == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0,
        )
        return response.content[0].text.strip()

    if provider == "gemini":
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.GenerationConfig(temperature=0),
            system_instruction=system_prompt,
        )
        return gen_model.generate_content(user_prompt).text.strip()


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a transcript editor. Your task: given a raw speech transcript, return \
the EXACT same text with every repetitive or false-start segment wrapped in \
square brackets [ ].

════════════════════════════════════════════════════════
WHAT TO MARK
════════════════════════════════════════════════════════

REPETITIONS / RESTARTS
  The speaker attempted a phrase, restarted, and eventually completed it.
  Mark ALL earlier incomplete attempts; keep only the LAST (most complete) version.

  Example:
    "Hi and welcome. Hi and welcome. Hi and welcome to a new lesson."
  →  "[Hi and welcome. Hi and welcome.] Hi and welcome to a new lesson."

FALSE STARTS
  The speaker began a sentence, stopped mid-way, and then said it properly.
  Mark only the incomplete attempt.

  Example:
    "And let's start it by going to our terminal and creating Let's start \
by going to our terminal and creating a folder."
  →  "[And let's start it by going to our terminal and creating] Let's start \
by going to our terminal and creating a folder."

EMBEDDED STUTTERS / RESTARTS
  Repetition fragments embedded inside a longer stretch of speech.
  Mark only the noisy block — from the exact point where speech becomes \
garbled/repetitive up to (but not including) the final clean continuation.

  Example:
    "And over here again, I'll make two folders. So I'll call And over and \
over and over here again, I'll make two folders one for the back end."
  →  "And over here again, I'll make two folders. [So I'll call And over and \
over and over here again, I'll make two folders] one for the back end."
  NOTE: "And over here again, I'll make two folders." is a clean, complete \
sentence — it is NOT part of the repetition and must stay outside the brackets.

SEMANTIC CORRECTIONS
  The speaker said something, then immediately corrected it with a more \
accurate or complete version. The first (incorrect) version should be marked, \
keeping only the corrected version.
  Look for: a word or phrase that names the wrong thing, followed by the \
speaker rephrasing with the right term — even if the surrounding sentence \
structure partially overlaps.

  Example:
    "And now let's open this up into our text ID. this folder into our AI \
coding ID for this module."
  →  "And now let's open [this up into our text ID.] this folder into our AI \
coding ID for this module."
  NOTE: "text ID" was the wrong term; "AI coding ID" is the correction. \
Mark the fragment containing the wrong term, keeping the corrected version.

════════════════════════════════════════════════════════
BRACKET PLACEMENT — CRITICAL RULES
════════════════════════════════════════════════════════
A. Place the OPENING bracket as LATE as possible.
   Only start the bracket at the exact word where the speech first becomes \
repetitive, garbled, or fragmented.
   Any clean, coherent content that precedes the problem — even if the same \
topic is discussed — must stay OUTSIDE the bracket.

B. Place the CLOSING bracket as EARLY as possible.
   End the bracket at the last word of the noisy/repeated content, right \
before the clean continuation begins.

C. A sentence that is complete and coherent on its own (ends with a period, \
question mark, etc.) must NEVER be pulled inside a bracket just because a \
repetition follows it.

════════════════════════════════════════════════════════
STRICT RULES
════════════════════════════════════════════════════════
1. Do NOT change, reorder, or omit any word outside the brackets.
2. Do NOT mark anything that, if removed, would break the coherence of what \
remains.
3. When a phrase restarts multiple times, keep only the LAST version and mark \
everything before it as a single block.
4. Preserve all whitespace and punctuation exactly as given — only add [ and ].
5. Your output must be the complete input text with brackets added. \
No explanation, no commentary, no extra text of any kind.

════════════════════════════════════════════════════════
VALIDATION
════════════════════════════════════════════════════════
Before replying, mentally remove every bracketed segment and re-read the result.
It must:
  ✓ Contain no repeated phrases in a row
  ✓ Be grammatically complete and make sense throughout
  ✓ Have no abandoned fragments
  ✓ Retain every clean sentence that existed before the repetition began

If it fails, adjust your brackets — especially by moving the opening bracket \
later — before replying.
"""


def mark_repetitions(text: str, model: str = "gpt-4o") -> str:
    """
    Mark repetitive / false-start segments in *text* by wrapping them in [ ].

    Removing every bracketed segment leaves behind coherent, well-structured
    English.  Uses an LLM — no algorithmic heuristics.

    Parameters
    ----------
    text  : Raw transcript text.
    model : LLM model name.  Provider is inferred from the name prefix:
              gpt-*/o1*/o3*/o4* → OpenAI   (OPENAI_API_KEY)
              claude-*          → Anthropic (ANTHROPIC_API_KEY)
              gemini-*          → Google    (GEMINI_API_KEY)

    Returns
    -------
    The original text with repetitive segments wrapped in square brackets.
    """
    provider = _infer_provider(model)
    print(f"[analyze_cuts] Sending text to {provider} ({model})...", file=sys.stderr)

    user_prompt = f"Mark the repetitive segments in the transcript below:\n\n{text}"
    result = _call_llm(_SYSTEM_PROMPT, user_prompt, model)

    print("[analyze_cuts] Done.", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mark repetitive/false-start segments in a transcript.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python analyze_cuts.py transcript.txt
  echo "hello hello world" | python analyze_cuts.py
  python analyze_cuts.py transcript.txt --ai-model claude-sonnet-4-6
  python analyze_cuts.py transcript.txt --ai-model gemini-1.5-pro --output marked.txt
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input text file (reads from stdin if omitted)",
    )
    parser.add_argument(
        "--ai-model",
        default="gpt-4o",
        metavar="MODEL",
        help=(
            "Model for analysis (default: gpt-4o). "
            "Provider is inferred from the model name: "
            "gpt-*/o1*/o3* → OpenAI (OPENAI_API_KEY), "
            "claude-* → Anthropic (ANTHROPIC_API_KEY), "
            "gemini-* → Google (GEMINI_API_KEY)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write result to this file instead of stdout",
    )

    args = parser.parse_args()

    if args.input:
        text = Path(args.input).read_text()
    else:
        if sys.stdin.isatty():
            print("Reading from stdin (Ctrl-D to finish)…", file=sys.stderr)
        text = sys.stdin.read()

    text = text.strip()
    if not text:
        print("Error: no input text provided.", file=sys.stderr)
        sys.exit(1)

    result = mark_repetitions(text, model=args.ai_model)

    if args.output:
        Path(args.output).write_text(result)
        print(f"[analyze_cuts] Output written to {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
