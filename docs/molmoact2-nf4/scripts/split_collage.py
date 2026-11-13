#!/usr/bin/env python
"""Split a saved two-camera collage into its halves, or inspect a collage.

The material under ``D:\\桌面\\萌芽杯\\policy_v1_refined`` stores the two camera
streams side by side.  The generating code is
``policy_v1_refined/bounded_run.py``::

    panel = cv2.cvtColor(np.hstack([row['images']['3'], row['images']['1']]),
                         cv2.COLOR_RGB2BGR)

so the JPEG is a plain horizontal concatenation whose halves are in visual
(left-to-right) order, even though the arrays that went in were RGB: the
RGB->BGR conversion happens once for the whole panel right before encoding.

And ``policy_v1_refined/live_probe.py`` labels the two slots::

    ('3', 'EXTERNAL / camera 3 -> model camera1'),
    ('1', 'WRIST / camera 1 -> model camera2')

Therefore:

    left  half  x in [0, W/2)  = camera '3' = external/overhead = camera1
    right half  x in [W/2, W)  = camera '1' = wrist             = camera2

This script performs that split without touching the camera pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--info", action="store_true")
    args = ap.parse_args()

    src = Path(args.input)
    img = Image.open(src).convert("RGB")
    w, h = img.size
    if w % 2:
        raise SystemExit(f"width {w} is not even; not a two-up collage?")
    half = w // 2

    left = img.crop((0, 0, half, h))
    right = img.crop((half, 0, w, h))
    Path(args.left).parent.mkdir(parents=True, exist_ok=True)
    left.save(args.left, quality=95)
    right.save(args.right, quality=95)

    if args.info:
        print(f"source {src} size={img.size} sha256={sha256(src)}")
        print(f"left  (overhead/camera1) -> {args.left} size={left.size}")
        print(f"right (wrist/camera2)    -> {args.right} size={right.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
