#!/usr/bin/env python
"""Synchronised read-only capture: two cameras + real joint positions.

Must be run with an interpreter whose executable lives on a medium-integrity
path (the LeRobot venv): DirectShow capture is refused to low-integrity
processes, which is what the project venv inherits from its folder label.

The arm is never commanded.  The serial TX path is wrapped by the existing
``readonly_bus.ReadOnlyPacketGuard`` (ping / read / sync-read only); torque is
read before and after and must be unchanged.

Each sample is one directory written atomically:

    <spool>/sample_000123/
        meta.json    seq, monotonic timestamps, image<->state skew, six joints
        top.jpg      overhead view (camera1)
        wrist.jpg    wrist view   (camera2)

``<spool>/latest.json`` always points at the newest completed sample, so a
consumer process can poll it.

Usage (LeRobot venv):
  python capture_live_readonly.py --duration 60 --hz 3 --spool tmp/live_spool
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
POLICY_DIR = Path(r"D:\桌面\萌芽杯\policy_v1_refined")
LEROBOT_SRC = Path(r"G:\LeRobot\source\lerobot-0.6.0\src")
CAL_PATH = Path(r"G:\LeRobot\data\calibration\robots\so_follower\lerobot.json")
REF_CAL_PATH = POLICY_DIR / "reference_calibration.json"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def open_camera(index: int, width: int, height: int, fps: int, warmup: int = 8):
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
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"camera {index} produced no frame")
    return cap, {"index": index, "height": int(frame.shape[0]), "width": int(frame.shape[1])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--camera-top", type=int, default=2)
    ap.add_argument("--camera-wrist", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hz", type=float, default=3.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--spool", default=str(HERE / "tmp" / "live_spool"))
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args()

    spool = Path(args.spool)
    if spool.exists():
        shutil.rmtree(spool)
    spool.mkdir(parents=True, exist_ok=True)

    status = {"created_at": now_iso(), "mode": "read-only capture", "status": "started"}
    status_path = spool / "capture_status.json"
    arm = None
    caps = {}
    try:
        sys.path.insert(0, str(LEROBOT_SRC))
        sys.path.insert(0, str(POLICY_DIR))
        from readonly_bus import ReadOnlyArm, validate_calibration

        calibration = validate_calibration(CAL_PATH, REF_CAL_PATH)
        cap_top, info_top = open_camera(args.camera_top, args.width, args.height, args.fps)
        cap_wrist, info_wrist = open_camera(args.camera_wrist, args.width, args.height, args.fps)
        caps = {"top": cap_top, "wrist": cap_wrist}

        arm = ReadOnlyArm(args.port, calibration)
        arm.connect()
        status.update(
            {
                "cameras": [info_top, info_wrist],
                "torque_before": arm.torque_before,
                "calibration_midpoints_ticks": {
                    n: (c["range_min"] + c["range_max"]) / 2 for n, c in calibration.items()
                },
            }
        )
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

        period = 1.0 / args.hz
        t_start = time.perf_counter()
        seq = 0
        written = 0
        while time.perf_counter() - t_start < args.duration:
            tick = time.perf_counter()
            grabs = {}
            for name, cap in caps.items():
                if not cap.grab():
                    raise RuntimeError(f"camera {name} grab failed")
                grabs[name] = time.perf_counter()
            frames = {}
            for name, cap in caps.items():
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise RuntimeError(f"camera {name} retrieve failed")
                frames[name] = frame
            state = arm.read()
            t_state = time.perf_counter()

            final = spool / f"sample_{seq:06d}"
            tmp = spool / f".tmp_{seq:06d}"
            tmp.mkdir(exist_ok=True)
            for name, frame in frames.items():
                cv2.imwrite(
                    str(tmp / f"{name}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
            meta = {
                "seq": seq,
                "t_wall": now_iso(),
                "elapsed_s": round(time.perf_counter() - t_start, 4),
                "t_grab_top": round(grabs["top"], 6),
                "t_grab_wrist": round(grabs["wrist"], 6),
                "t_state": round(t_state, 6),
                "image_state_skew_ms": round((t_state - max(grabs.values())) * 1000, 2),
                "view_skew_ms": round((grabs["wrist"] - grabs["top"]) * 1000, 2),
                "state": [round(float(x), 4) for x in state],
                "state_raw_ticks_note": "six joint values in the LeRobot DEGREES / 0-100 convention",
            }
            (tmp / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            os.replace(tmp, final)
            # No pointer file: on Windows a reader holding "latest.json" open
            # makes os.replace fail (share-delete), and the consumer can simply
            # scan for the highest sample_* directory instead.
            written += 1
            seq += 1
            if seq % 15 == 0:
                print(
                    f"seq={seq} elapsed={time.perf_counter()-t_start:6.2f}s "
                    f"skew={meta['image_state_skew_ms']:5.1f}ms state={meta['state']}",
                    flush=True,
                )
            sleep_for = period - (time.perf_counter() - tick)
            if sleep_for > 0:
                time.sleep(sleep_for)
        status["status"] = "completed"
        status["samples"] = written
    except Exception:
        status["status"] = "failed"
        status["error"] = traceback.format_exc()
        raise
    finally:
        if arm is not None:
            try:
                arm.close()
                status["bus"] = arm.report()
                status["torque_after"] = arm.torque_after
                status["torque_unchanged"] = arm.torque_before == arm.torque_after
                status["allowed_packets"] = arm.report().get("allowed_packet_counts")
                status["blocked_packets"] = arm.report().get("blocked_packets")
            except Exception as e:  # noqa: BLE001
                status["arm_close_error"] = f"{type(e).__name__}: {e}"
        for cap in caps.values():
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps({k: status.get(k) for k in
                          ("status", "samples", "blocked_packets", "torque_unchanged")}))
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
