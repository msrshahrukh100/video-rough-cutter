# video-rough-cutter

Automatically clean up rough video recordings by removing speech disfluencies — false starts, progressive restarts, mid-sentence stutters, semantic corrections, filler-only utterances, and long silences. The result stitches together as if the speaker said it perfectly the first time.

Designed for lesson recordings and screencasts where you want a clean cut without sitting in a video editor.

## How it works

The pipeline has three stages:

```
Input Video
  │
  ▼
[1] Transcribe        openai-whisper → scratch/<stem>/transcript.json
  │                   (word-level timestamps)
  ▼
[2] Detect Cuts       LLM or algorithm → scratch/<stem>/cuts.json
  │                   (keep intervals)
  ▼
[3] Apply Cuts        ffmpeg filter_complex → output video
                      (H.264 / AAC re-encode)
```

**Stage 1 — Transcribe:** Whisper produces a word-level transcript with precise start/end timestamps for every word.

**Stage 2 — Detect Cuts:** Words are grouped into utterances by silence gaps (≥ 0.4 s). An LLM (or a fallback algorithm) analyses the utterances and returns three things to cut:
- `cuts` — repetition/restart chains (keeps only the final, complete attempt)
- `remove` — filler-only utterances with no real content ("um", "uh", "so like you know")
- `trim_silence_after` — gaps between utterances that feel dead (≥ ~2 s), trimmed down to 0.4 s

**Stage 3 — Apply Cuts:** FFmpeg builds a `filter_complex` that trims and concatenates exactly the keep intervals, then re-encodes to H.264 (CRF 18, fast preset) + AAC.

## Requirements

**System dependencies** (install separately):
- Python 3.9+
- `ffmpeg` (includes `ffprobe`) — e.g. `brew install ffmpeg`

**API keys** — only needed for AI mode (the default). Set in your environment:

| Model prefix | Provider | Environment variable |
|---|---|---|
| `gpt-*`, `o1*`, `o3*`, `o4*` | OpenAI | `OPENAI_API_KEY` |
| `claude-*` | Anthropic | `ANTHROPIC_API_KEY` |
| `gemini-*` | Google | `GEMINI_API_KEY` |

## Setup

```bash
# Clone and set up the virtual environment
git clone <repo-url>
cd video-rough-cutter
./setup.sh
```

`setup.sh` creates `.venv/` with Python 3.9, upgrades pip, and installs `requirements.txt` (Whisper, PyTorch, OpenAI, Anthropic, and Google Generative AI SDKs).

## Usage

```bash
./cut.sh <input_video> [options]
```

Output is saved as `<stem>_cut.mp4` next to the input file unless `--output` is specified.

### Examples

```bash
# Full pipeline — transcribe + AI detection + apply cuts
./cut.sh lecture.mp4

# Use a specific Whisper model (tiny/base/small/medium/large)
./cut.sh lecture.mp4 --model small

# Use a different AI model for cut detection
./cut.sh lecture.mp4 --ai-model claude-opus-4-6
./cut.sh lecture.mp4 --ai-model gemini-2.5-pro

# Tune detection without re-running Whisper (fast iteration)
./cut.sh lecture.mp4 --skip-transcribe

# Re-apply cuts after manually editing cuts.json
./cut.sh lecture.mp4 --skip-transcribe --skip-detect

# No API key needed — greedy exact word-match algorithm
./cut.sh lecture.mp4 --no-ai

# Save output to a specific path
./cut.sh lecture.mp4 --output /tmp/lecture_clean.mp4
```

### All options

| Option | Default | Description |
|---|---|---|
| `--model` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `--ai-model` | `gpt-4o` | LLM for cut detection (provider inferred from name) |
| `--no-ai` | off | Use greedy word-match algorithm instead of an LLM |
| `--min-match` | `2` | `--no-ai` only: minimum consecutive words for a match |
| `--padding` | `0.08` | Seconds of buffer trimmed before each cut point |
| `--output` | `<stem>_cut.mp4` | Output file path |
| `--scratch-dir` | `./scratch` | Directory for intermediate files |
| `--skip-transcribe` | off | Reuse existing `transcript.json` |
| `--skip-detect` | off | Reuse existing `cuts.json` |

## Intermediate files

Both intermediate files are saved under `scratch/<video-stem>/` and are gitignored.

**`transcript.json`**
```json
{
  "words": [
    { "word": "hello", "start": 0.0, "end": 0.5 },
    ...
  ],
  "text": "hello and welcome to..."
}
```

**`cuts.json`**
```json
{
  "keep_intervals": [
    { "start": 6.92, "end": 13.74 },
    ...
  ],
  "cuts_made": 41,
  "total_duration": 839.38
}
```

You can manually edit `cuts.json` to add, remove, or adjust intervals, then re-run with `--skip-transcribe --skip-detect` to apply your changes.

## Choosing a Whisper model

| Model | Speed | Accuracy |
|---|---|---|
| `tiny` | Very fast | Low — suitable only for testing |
| `base` | Fast | Good enough for clear audio |
| `small` | Moderate | Better for accented speech |
| `medium` | Slow | High quality |
| `large` | Very slow | Best accuracy |

`base` is the default and works well for most screencasts recorded in a quiet environment.

## AI mode vs. algorithm mode

**AI mode** (default) sends utterances to an LLM with a detailed prompt that identifies five cut types. It handles natural speech imperfectly — e.g. "let's open our IDE" → "let's open our AI coding IDE" (semantic correction) — and requires an API key.

**Algorithm mode** (`--no-ai`) uses a greedy forward scan for consecutive exact word matches. It is free, fast, and requires no API key, but only catches verbatim repetitions and misses semantic corrections, filler utterances, and silence.

## Running the algorithm unit test

```bash
python src/detect_cuts.py
```

No network or API key required.
