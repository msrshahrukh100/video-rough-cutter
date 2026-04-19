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

The default LLM model is `gpt-5.4` (OpenAI). Provider is inferred from the `--ai-model` flag:
- `gpt-*` / `o1*` / `o3*` / `o4*` → OpenAI (`OPENAI_API_KEY`)
- `claude-*` → Anthropic (`ANTHROPIC_API_KEY`)
- `gemini-*` → Google (`GEMINI_API_KEY`)

## Common Commands

```bash
# Full pipeline with specific Whisper model
./cut.sh video.mp4 --model base

# Use a different LLM provider
./cut.sh video.mp4 --ai-model claude-sonnet-4-6
./cut.sh video.mp4 --ai-model gemini-1.5-pro

# Skip transcription (reuse existing transcript.json)
./cut.sh video.mp4 --skip-transcribe

# Skip transcription + marking (reuse existing marked.txt)
./cut.sh video.mp4 --skip-transcribe --skip-mark

# Generate cuts.json only — review/edit before cutting
./cut.sh video.mp4 --skip-apply

# Re-apply cuts after manually editing cuts.json
./cut.sh video.mp4 --skip-transcribe --skip-detect

# Custom output path and padding
./cut.sh video.mp4 --output /path/to/output.mp4 --padding 0.1

# Run timestamp-alignment unit tests (no network needed)
python src/detect_cuts.py

# Standalone: mark repetitions in a transcript text file
python src/analyze_cuts.py transcript.txt
python src/analyze_cuts.py transcript.txt --ai-model claude-sonnet-4-6

# Helper scripts
python src/scripts/transcript_to_text.py scratch/<stem>/transcript.json
python src/scripts/remove_marked_segments.py scratch/<stem>/marked.txt
```

## Architecture

Four-stage pipeline orchestrated by `src/pipeline.py`:

```
Input Video
    → [src/transcribe.py]   → scratch/<stem>/transcript.json   (Whisper word timestamps)
    → [src/analyze_cuts.py] → scratch/<stem>/marked.txt        (LLM marks [bad segments])
    → [src/detect_cuts.py]  → scratch/<stem>/cuts.json         (timestamp alignment → keep intervals)
    → [src/apply_cuts.py]   → Output Video                     (FFmpeg re-encode)
```

**`src/pipeline.py`** — CLI entry point. The `--skip-*` flags map to stages: `--skip-transcribe`, `--skip-mark`, `--skip-detect`, `--skip-apply`. `--skip-detect` implies skipping mark too. Scratch files live under `scratch/<input-stem>/`.

**`src/transcribe.py`** — Loads a Whisper model and extracts word-level timestamps into `transcript.json` as `{words: [{word, start, end}], text: "..."}`. Words are lowercased and stripped.

**`src/analyze_cuts.py`** — Sends the raw transcript text to an LLM with a detailed system prompt instructing it to wrap repetitive/false-start segments in square brackets `[...]`. The invariant is: removing every `[...]` block leaves coherent English. Supports OpenAI, Anthropic, and Google Gemini. Can also be run standalone as a CLI (reads from a file or stdin). The `--output` flag writes `marked.txt` to a file.

**`src/detect_cuts.py`** — Parses the `[...]` markers in `marked.txt` and aligns them to word timestamps from `transcript.json` using `difflib.SequenceMatcher` (with `autojunk=False` — critical because common filler words like "and", "so", "the" would otherwise be skipped). Inverts the cut ranges into keep intervals. The `--padding` flag (default `0.08s`) trims a small buffer before each cut point. Running this file directly executes unit tests.

**`src/apply_cuts.py`** — Builds an FFmpeg `filter_complex` with trim/concat operations and re-encodes to H.264 (CRF 18, fast preset) + AAC.

### Manual editing workflow

The `scratch/<stem>/marked.txt` file can be hand-edited between the mark and detect stages. The format is the plain transcript text with `[cut this out]` brackets added. Use `--skip-transcribe --skip-mark` to re-run only timestamp alignment and apply after editing.

The `scratch/` directory is gitignored; video files are also gitignored.
