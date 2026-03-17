"""Orchestrator: transcribe → mark → detect cuts → apply cuts."""

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
  # Full pipeline (requires an LLM API key)
  ./cut.sh my_video.mp4 --model base

  # Re-run marking + timestamp mapping without re-transcribing
  ./cut.sh my_video.mp4 --skip-transcribe

  # Manually edit scratch/<stem>/marked.txt, then re-map timestamps only
  ./cut.sh my_video.mp4 --skip-transcribe --skip-mark

  # Just re-apply cuts (e.g. after editing cuts.json manually)
  ./cut.sh my_video.mp4 --skip-transcribe --skip-detect
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

    # LLM options
    parser.add_argument(
        "--ai-model",
        default="gpt-4o",
        metavar="MODEL",
        help=(
            "Model for text marking (default: gpt-4o). "
            "Provider is inferred from the model name: "
            "gpt-*/o1*/o3* → OpenAI (OPENAI_API_KEY), "
            "claude-* → Anthropic (ANTHROPIC_API_KEY), "
            "gemini-* → Google (GEMINI_API_KEY)."
        ),
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
        "--skip-mark",
        action="store_true",
        help="Reuse existing marked.txt (skip LLM marking step)",
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
    marked_path = scratch_dir / "marked.txt"
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

    # Step 2: Detect cuts (mark → align timestamps)
    if args.skip_detect:
        if not cuts_path.exists():
            print(
                f"Error: --skip-detect set but {cuts_path} not found.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[pipeline] Skipping cut detection, using {cuts_path}")
    else:
        if args.skip_mark:
            if not marked_path.exists():
                print(
                    f"Error: --skip-mark set but {marked_path} not found.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"[pipeline] Skipping marking step, using {marked_path}")

        print("[pipeline] Getting video duration via ffprobe...")
        total_duration = get_duration(str(input_path))
        print(f"[pipeline] Duration: {total_duration:.2f}s")

        from detect_cuts import run as run_detect
        run_detect(
            str(transcript_path),
            str(cuts_path),
            total_duration=total_duration,
            padding=args.padding,
            model=args.ai_model,
            marked_path=str(marked_path),
        )

    # Step 3: Apply cuts
    from apply_cuts import apply_cuts
    apply_cuts(str(input_path), str(cuts_path), str(output_path))

    print(f"\n[pipeline] Done! Output: {output_path}")


if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    main()
