"""Generate synthetic alert-lamp snapshots for manual /v1/alert-blink testing.

Two scenarios are written:

* ``blink/``   - six frames sampled from a known 1000 ms blinker
                 (lit half 0..499, off half 500..999, phase 0), including one
                 frame whose sampling window straddles a switch;
* ``steady/``  - four always-lit frames (the "was every lucky shot lit?" case),
                 plus a ``request.json`` sidecar describing the multipart form
                 fields and a ready-to-paste curl command.

Usage:
    python scripts/make_blink_sample.py --out samples/blink_demo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

WIDTH, HEIGHT = 128, 128
BOX = (44, 44, 40, 40)  # x, y, width, height
RED = (210, 35, 35)
DARK = (25, 25, 30)


def _frame(lit: bool, path: Path) -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), (60, 60, 66))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 88, 88], outline=(180, 180, 190), width=2)  # lamp housing
    d.ellipse([BOX[0], BOX[1], BOX[0] + BOX[2] - 1, BOX[1] + BOX[3] - 1],
              fill=RED if lit else DARK)
    img.save(path, format="PNG")


def _lamp_lit_at(t: int, period: int, phase: int) -> bool:
    return (t - phase) % period < period // 2


def build_blink_scenario(out: Path) -> dict:
    d = out / "blink"
    d.mkdir(parents=True, exist_ok=True)
    period, phase = 1000, 0
    # (filename, t_start, t_end); each window contains at least one lit/off
    # instant matching the observed state; frame 4 straddles a switch.
    spec = [
        ("frame1.png", 0, 0),        # lit (switch at 500 not reached)
        ("frame2.png", 300, 300),    # lit
        ("frame3.png", 600, 600),    # off
        ("frame4.png", 450, 550),    # window straddles t=500; observed lit
        ("frame5.png", 1600, 1600),  # off (600 ms into period 2)
        ("frame6.png", 2100, 2100),  # lit (100 ms into period 3)
    ]
    meta = {"lamp_box": list(BOX), "period_ms": period, "phase_ms": phase, "frames": []}
    for name, t_start, t_end in spec:
        lit = _lamp_lit_at(t_start, period, phase)
        _frame(lit, d / name)
        meta["frames"].append({"file": name, "t_start_ms": t_start,
                               "t_end_ms": t_end, "observed": "lit" if lit else "off"})
    (d / "request.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def build_steady_scenario(out: Path) -> dict:
    d = out / "steady"
    d.mkdir(parents=True, exist_ok=True)
    spec = [
        ("frame1.png", 0, 0),
        ("frame2.png", 700, 700),
        ("frame3.png", 1400, 1400),
        ("frame4.png", 2100, 2100),
    ]
    meta = {"lamp_box": list(BOX), "frames": []}
    for name, t_start, t_end in spec:
        _frame(True, d / name)
        meta["frames"].append({"file": name, "t_start_ms": t_start,
                               "t_end_ms": t_end, "observed": "lit"})
    (d / "request.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples/blink_demo", type=Path)
    args = parser.parse_args()
    blink = build_blink_scenario(args.out)
    steady = build_steady_scenario(args.out)
    print(f"blink scenario ({len(blink['frames'])} frames, ground-truth "
          f"T={blink['period_ms']}ms, phase={blink['phase_ms']}ms): "
          f"{(args.out / 'blink').resolve()}")
    print(f"steady scenario ({len(steady['frames'])} always-lit frames): "
          f"{(args.out / 'steady').resolve()}")


if __name__ == "__main__":
    main()
