"""Remove segments wrapped in square brackets from marked transcript text.

The input is the output of analyze_cuts.py — plain text with repetitive /
false-start segments wrapped in [ ].  This script strips every bracketed
segment (including surrounding whitespace) and writes the clean result to a
file or stdout.

Usage
-----
  python remove_marked_segments.py marked.txt
  python remove_marked_segments.py marked.txt --output clean.txt
  echo "[bad part] good part" | python remove_marked_segments.py
"""

import argparse
import re
import sys
from pathlib import Path


def remove_marked_segments(text: str) -> str:
    """
    Strip every [...] segment from *text* and return the cleaned string.

    Handles:
    - Single-line and multi-line brackets
    - Extra whitespace left behind after removal (collapsed to one space)
    - Leading/trailing whitespace
    """
    # Remove every [...] block (non-greedy, DOTALL for multi-line spans)
    cleaned = re.sub(r"\[.*?\]", "", text, flags=re.DOTALL)

    # Collapse runs of whitespace (spaces/tabs) to a single space,
    # but preserve intentional newlines (paragraph breaks).
    lines = cleaned.splitlines()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]

    # Drop lines that became empty after stripping, then rejoin
    cleaned = "\n".join(line for line in lines if line)

    return cleaned.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove [bracketed] segments from a marked transcript.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python remove_marked_segments.py marked.txt
  python remove_marked_segments.py marked.txt --output clean.txt
  echo "[bad] good part" | python remove_marked_segments.py
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Marked text file (reads from stdin if omitted)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write cleaned text to this file (default: print to stdout)",
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

    result = remove_marked_segments(text)

    if args.output:
        Path(args.output).write_text(result)
        print(f"Cleaned text written to {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()
