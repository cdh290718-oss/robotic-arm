#!/usr/bin/env python
"""Camera-only frame spooler for the closed-loop run.

Must run under an interpreter on a medium-integrity path (LeRobot venv):
DirectShow refuses to open cameras for low-integrity processes.

Writes one directory per frame pair, atomically, so a consumer can always pick
the newest complete pair:

    <spool>/frame_000123/{top.jpg,wrist.jpg,meta.json}
    <spool>/status.json      {"status": "...", "frames": N}

It never touches the serial port, so it cannot conflict with the controller.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import cv2


def open_camera(index: int, width: int, height: int, fps: int, warmup: int = 6):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"camera {index} cannot be opened")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(warmup):
        cap.read()
    return cap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-top", type=int, default=2)
    ap.add_argument("--camera-wrist", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hz", type=float, default=8.0)
    ap.add_argument("--duration", type=float, default=200.0)
    ap.add_argument("--spool", default=str(Path(__file__).resolve().parent / "tmp" / "frames"))
    ap.add_argument("--quality", type=int, default=88)
    args = ap.parse_args()

    spool = Path(args.spool)
    if spool.exists():
        shutil.rmtree(spool)
    spool.mkdir(parents=True, exist_ok=True)
    status = {"status": "starting", "frames": 0, "started": time.time()}

    def write_status(**kw):
        status.update(kw)
        (spool / "status.json").write_text(json.dumps(status), encoding="utf-8")

    caps = {}
    try:
        caps["top"] = open_camera(args.camera_top, args.width, args.height, args.fps)
        caps["wrist"] = open_camera(args.camera_wrist, args.width, args.height, args.fps)
        write_status(status="running")
        period = 1.0 / args.hz
        t_start = time.perf_counter()
        seq = 0
        while time.perf_counter() - t_start < args.duration:
            tick = time.perf_counter()
            stamps = {}
            for name, cap in caps.items():
                if not cap.grab():
                    raise RuntimeError(f"{name} grab failed")
                stamps[name] = time.time()
            final = spool / f"frame_{seq:06d}"
            tmp = spool / f".tmp_{seq:06d}"
            tmp.mkdir(exist_ok=True)
            for name, cap in caps.items():
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise RuntimeError(f"{name} retrieve failed")
                cv2.imwrite(str(tmp / f"{name}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            (tmp / "meta.json").write_text(json.dumps({
                "seq": seq,
                "t_wall": stamps["top"],
                "t_top": stamps["top"],
                "t_wrist": stamps["wrist"],
                "view_skew_ms": round((stamps["wrist"] - stamps["top"]) * 1000, 2),
            }), encoding="utf-8")
            os.replace(tmp, final)
            seq += 1
            if seq % 80 == 0:
                write_status(status="running", frames=seq)
            sleep = period - (time.perf_counter() - tick)
            if sleep > 0:
                time.sleep(sleep)
        write_status(status="completed", frames=seq)
        print(json.dumps({"status": "completed", "frames": seq}), flush=True)
    except Exception as e:  # noqa: BLE001
        write_status(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        for cap in caps.values():
            cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
