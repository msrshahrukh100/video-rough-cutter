"""Step 3: Apply keep intervals to video using ffmpeg filter_complex."""

import json
import shutil
import subprocess
import sys
from pathlib import Path


def apply_cuts(input_path: str, cuts_path: str, output_path: str) -> None:
    data = json.loads(Path(cuts_path).read_text())
    keep_intervals = data["keep_intervals"]
    cuts_made = data["cuts_made"]

    if cuts_made == 0:
        print("[apply_cuts] No cuts detected — copying input to output.")
        shutil.copy2(input_path, output_path)
        return

    n = len(keep_intervals)
    print(f"[apply_cuts] Applying {cuts_made} cut(s), keeping {n} interval(s)...")

    filter_parts = []
    for idx, interval in enumerate(keep_intervals):
        s, e = interval["start"], interval["end"]
        filter_parts.append(
            f"[0:v]trim=start={s:.6f}:end={e:.6f},setpts=PTS-STARTPTS[v{idx}];"
            f"[0:a]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS[a{idx}]"
        )

    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
    filter_complex = (
        ";\n".join(filter_parts)
        + f";\n{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]"
    )

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac",
        output_path,
    ]

    print(f"[apply_cuts] Running ffmpeg...")
    subprocess.run(cmd, check=True)
    print(f"[apply_cuts] Output → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: apply_cuts.py <input_video> <cuts_json> <output_video>")
        sys.exit(1)
    apply_cuts(sys.argv[1], sys.argv[2], sys.argv[3])
