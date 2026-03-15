# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and Running

```bash
# Initial setup (creates .venv with Python 3.9)
./setup.sh

# Run the full pipeline
./cut.sh <input_video.mp4> [options]
```

Requires `ffmpeg`/`ffprobe` installed externally (not in requirements.txt).

AI mode (default) requires `OPENAI_API_KEY` in the environment.

## Common Commands

```bash
# Full pipeline with specific Whisper model
./cut.sh video.mp4 --model base

# Skip transcription (reuse existing transcript.json)
./cut.sh video.mp4 --skip-transcribe

# Skip transcription and detection (re-apply cuts after manually editing cuts.json)
./cut.sh video.mp4 --skip-transcribe --skip-detect

# No AI — use greedy word-match algorithm instead (no API key needed)
./cut.sh video.mp4 --no-ai

# Custom output path
./cut.sh video.mp4 --output /path/to/output.mp4

# Run algorithmic unit tests (no network needed)
python src/detect_cuts.py
```

## Architecture

Three-stage pipeline orchestrated by `src/pipeline.py`:

```
Input Video
    → [src/transcribe.py]   → scratch/<stem>/transcript.json   (Whisper word timestamps)
    → [src/detect_cuts.py]  → scratch/<stem>/cuts.json         (keep intervals)
    → [src/apply_cuts.py]   → Output Video                     (FFmpeg re-encode)
```

**`src/pipeline.py`** — CLI entry point. Parses args, validates input, creates scratch directory, and conditionally runs each stage. `--skip-transcribe` and `--skip-detect` flags allow partial re-runs for tuning.

**`src/transcribe.py`** — Loads a Whisper model and extracts word-level timestamps into `transcript.json` as `{words: [{word, start, end}], text: "..."}`.

**`src/detect_cuts.py`** — Identifies segments to cut. Two modes:
- **AI mode** (default): Groups words into utterances (silence gap ≥ 0.4s), sends them to GPT-4o with a detailed prompt to identify 5 cut types (progressive restarts, mid-sentence stutters, semantic corrections, filler-only utterances, long silences).
- **Algorithm mode** (`--no-ai`): Greedy forward scan for exact consecutive word matches (minimum 2 words).

Outputs `cuts.json` as `{keep_intervals: [{start, end}], cuts_made: N, total_duration: S}`.

**`src/apply_cuts.py`** — Builds an FFmpeg `filter_complex` with trim/concat operations to re-encode to H.264 (CRF 18) + AAC.

The `scratch/` directory is gitignored; video files are also gitignored.
