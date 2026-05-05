"""
Compose an Isaac viewport recording side-by-side with the DS interaction video.

Typical workflow:
  1. Record the Isaac window while running deploy_dual_arm.py.
  2. Generate the DS interaction video from the diagnostic pickle.
  3. Use this script to stack them into one comparison video.

Example:
  python scripts/animate_lpvds_interaction.py \\
    --diag data/results/lpvds_interaction.pkl \\
    --out data/results/ds_interaction_radial.mp4 \\
    --radial_field

  python scripts/compose_isaac_ds_video.py \\
    --isaac data/results/isaac_render.mp4 \\
    --ds data/results/ds_interaction_radial.mp4 \\
    --out data/results/isaac_vs_ds_interaction.mp4
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def run(cmd):
    print("[CMD]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac", required=True,
                        help="Recorded Isaac viewport video")
    parser.add_argument("--ds", required=True,
                        help="DS interaction animation video")
    parser.add_argument("--out", default="data/results/isaac_vs_ds_interaction.mp4")
    parser.add_argument("--height", type=int, default=720,
                        help="Common output panel height")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--label", action="store_true",
                        help="Overlay simple panel labels")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for video composition.")

    isaac = Path(args.isaac)
    ds = Path(args.ds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not isaac.exists():
        raise FileNotFoundError(isaac)
    if not ds.exists():
        raise FileNotFoundError(ds)

    left = (
        f"[0:v]fps={args.fps},scale=-2:{args.height},"
        f"setsar=1"
    )
    right = (
        f"[1:v]fps={args.fps},scale=-2:{args.height},"
        f"setsar=1"
    )
    if args.label:
        left += (
            ",drawtext=text='Isaac render':x=24:y=24:"
            "fontsize=32:fontcolor=white:box=1:boxcolor=black@0.45"
        )
        right += (
            ",drawtext=text='DS + modulation':x=24:y=24:"
            "fontsize=32:fontcolor=white:box=1:boxcolor=black@0.45"
        )
    filt = f"{left}[l];{right}[r];[l][r]hstack=inputs=2[v]"

    run([
        "ffmpeg", "-y",
        "-i", str(isaac),
        "-i", str(ds),
        "-filter_complex", filt,
        "-map", "[v]",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "veryfast",
        str(out),
    ])
    print(f"[VIDEO] Saved {out}")


if __name__ == "__main__":
    main()
