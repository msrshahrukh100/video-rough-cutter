"""Orchestrator: transcribe → detect cuts → apply cuts."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def get_duration(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove speech repetitions/false-starts from a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Full pipeline (AI-based cut detection, requires OPENAI_API_KEY)
  ./cut.sh my_video.mp4 --model base

  # Tune detection without re-transcribing
  ./cut.sh my_video.mp4 --skip-transcribe

  # Just re-apply cuts (e.g. after editing cuts.json manually)
  ./cut.sh my_video.mp4 --skip-transcribe --skip-detect

  # Use algorithmic mode (no API key needed, exact word matches only)
  ./cut.sh my_video.mp4 --no-ai
        """,
    )
    parser.add_argument("input_video", help="Path to input video file")

    # Whisper options
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help=(
            "Whisper model (default: base). "
            "tiny/base=fast; small/medium=production accuracy; large=best, slow"
        ),
    )

    # AI detection options
    ai_group = parser.add_argument_group("AI cut detection (default mode)")
    ai_group.add_argument(
        "--ai-model",
        default="gpt-4o",
        metavar="MODEL",
        help="OpenAI model for cut detection (default: gpt-4o). Requires OPENAI_API_KEY.",
    )
    ai_group.add_argument(
        "--no-ai",
        action="store_true",
        help=(
            "Use algorithmic word-match detection instead of OpenAI. "
            "Faster and free, but only catches exact word repetitions."
        ),
    )

    # Algorithm fallback options (only relevant with --no-ai)
    alg_group = parser.add_argument_group("Algorithmic mode options (--no-ai only)")
    alg_group.add_argument(
        "--min-match",
        type=int,
        default=2,
        metavar="INT",
        help="Min consecutive words that must match to detect a restart (default: 2)",
    )

    # Shared options
    parser.add_argument(
        "--padding",
        type=float,
        default=0.08,
        metavar="FLOAT",
        help="Seconds of buffer trimmed before each cut point (default: 0.08)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output file path (default: <stem>_cut.mp4 next to input)",
    )
    parser.add_argument(
        "--scratch-dir",
        default="./scratch",
        metavar="PATH",
        help="Directory for temporary files (default: ./scratch)",
    )
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Reuse existing transcript.json (skip Whisper)",
    )
    parser.add_argument(
        "--skip-detect",
        action="store_true",
        help="Reuse existing cuts.json (skip cut detection)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_video).resolve()
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    stem = input_path.stem
    scratch_dir = Path(args.scratch_dir) / stem
    scratch_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = scratch_dir / "transcript.json"
    cuts_path = scratch_dir / "cuts.json"
    output_path = (
        Path(args.output) if args.output
        else input_path.parent / f"{stem}_cut.mp4"
    )

    # Step 1: Transcribe
    if args.skip_transcribe:
        if not transcript_path.exists():
            print(
                f"Error: --skip-transcribe set but {transcript_path} not found.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[pipeline] Skipping transcription, using {transcript_path}")
    else:
        from transcribe import transcribe
        transcribe(str(input_path), args.model, str(transcript_path))

    # Step 2: Detect cuts
    if args.skip_detect:
        if not cuts_path.exists():
            print(
                f"Error: --skip-detect set but {cuts_path} not found.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[pipeline] Skipping cut detection, using {cuts_path}")
    else:
        print("[pipeline] Getting video duration via ffprobe...")
        total_duration = get_duration(str(input_path))
        print(f"[pipeline] Duration: {total_duration:.2f}s")

        from detect_cuts import run as run_detect
        run_detect(
            str(transcript_path),
            str(cuts_path),
            total_duration=total_duration,
            padding=args.padding,
            use_ai=not args.no_ai,
            openai_model=args.ai_model,
            min_match=args.min_match,
        )

    # Step 3: Apply cuts
    from apply_cuts import apply_cuts
    apply_cuts(str(input_path), str(cuts_path), str(output_path))

    print(f"\n[pipeline] Done! Output: {output_path}")


if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    main()
