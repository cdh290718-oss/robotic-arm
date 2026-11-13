#!/usr/bin/env python
"""Live pose guide: read the arm and show how far each joint is from the usable window.

You rotate the arm by hand (torque is off).  This program only reads
``Present_Position`` through the existing read-only packet guard (ping / read /
sync-read only — it cannot write a goal, a gain, a lock or a calibration).

For every joint it shows

    current angle | model-frame angle | usable window | aim-for range | what to do

where the usable window is the checkpoint's state q01..q99 taken through the
official joint-frame bridge (``model = signs*arm + offsets``) and intersected
with this arm's calibrated travel.  The "aim for" range is that window shrunk
by a safety margin, so the model's output is not sitting on its own limit.

Usage:
  python guide_pose_window.py
  python guide_pose_window.py --hz 4 --margin 10 --gripper-margin 5
  python guide_pose_window.py --plain --seconds 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POLICY_DIR = Path(r"D:\桌面\萌芽杯\policy_v1_refined")
LEROBOT_SRC = Path(r"G:\LeRobot\source\lerobot-0.6.0\src")
CAL_PATH = Path(r"G:\LeRobot\data\calibration\robots\so_follower\lerobot.json")
REF_CAL_PATH = POLICY_DIR / "reference_calibration.json"
NORM_TAG = "so100_so101_molmoact2"
MAX_RES = 4095

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SHORT = ("pan", "lift", "elbow", "wrist", "roll", "grip")
UNITS = ("deg", "deg", "deg", "deg", "deg", "%")
PASSIVE = {4}  # wrist_roll: passive joint, not commanded

GREEN, RED, YELLOW, DIM, RESET = "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[2m", "\x1b[0m"


def load_windows(model_dir: Path, calibration: dict, margin: float, gripper_margin: float):
    sys.path.insert(0, str(HERE / "coordinate_bridge_20261113"))
    from joint_frame_bridge import model_to_arm

    norm = json.loads((model_dir / "norm_stats.json").read_text(encoding="utf-8"))
    st = norm["metadata_by_tag"][NORM_TAG]["state_stats"]
    q01 = np.asarray(st["q01"], dtype=np.float64)
    q99 = np.asarray(st["q99"], dtype=np.float64)
    s_lo = np.minimum(model_to_arm(q01), model_to_arm(q99))
    s_hi = np.maximum(model_to_arm(q01), model_to_arm(q99))

    travel_lo, travel_hi = np.zeros(6), np.zeros(6)
    for i, name in enumerate(NAMES):
        c = calibration[name]
        if name == "gripper":
            travel_lo[i], travel_hi[i] = 0.0, 100.0
        else:
            mid = (c["range_min"] + c["range_max"]) / 2
            travel_lo[i] = (c["range_min"] - mid) * 360 / MAX_RES
            travel_hi[i] = (c["range_max"] - mid) * 360 / MAX_RES

    lo = np.maximum(s_lo, travel_lo)
    hi = np.minimum(s_hi, travel_hi)
    goal_lo, goal_hi = lo.copy(), hi.copy()
    for i in range(6):
        m = gripper_margin if i == 5 else margin
        a, b = lo[i] + m, hi[i] - m
        if a > b:  # window narrower than the margin: aim at its centre
            a = b = (lo[i] + hi[i]) / 2
        goal_lo[i], goal_hi[i] = a, b
    return lo, hi, goal_lo, goal_hi, travel_lo, travel_hi


def render(local, model, lo, hi, gl, gh, t_lo, t_hi, margin, color) -> tuple[str, bool]:
    def paint(text, code):
        return f"{code}{text}{RESET}" if color else text

    rows = []
    head = (
        f"{'joint':14s}{'now':>9s}{'model':>9s}"
        f"{'usable window':>20s}{'aim for':>20s}   action"
    )
    rows.append(head)
    rows.append("-" * 96)
    ready = 0
    controllable = 0
    todo = []
    for i, name in enumerate(NAMES):
        v = float(local[i])
        unit = UNITS[i]
        inside = lo[i] - 1e-9 <= v <= hi[i] + 1e-9
        in_goal = gl[i] - 1e-9 <= v <= gh[i] + 1e-9

        if i in PASSIVE:
            act = paint("passive joint (not commanded)", DIM)
            status_ok = True
        else:
            controllable += 1
            if in_goal:
                act = paint("ok", GREEN)
                status_ok = True
                ready += 1
            elif v < gl[i]:
                act = paint(f"increase by {gl[i] - v:6.2f} {unit}", RED)
                todo.append(f"{SHORT[i]} +{gl[i] - v:.1f}")
                status_ok = False
            else:
                act = paint(f"decrease by {v - gh[i]:6.2f} {unit}", RED)
                todo.append(f"{SHORT[i]} -{v - gh[i]:.1f}")
                status_ok = False
            if not inside:
                act += paint("  [OUTSIDE]", RED)
            elif not in_goal:
                act += paint("  (near edge)", YELLOW)

        if i != 5:
            remain_lo, remain_hi = v - t_lo[i], t_hi[i] - v
            if remain_lo < 3.0:
                act += paint(f"  travel_end {remain_lo:.1f}", YELLOW)
            elif remain_hi < 3.0:
                act += paint(f"  travel_end {remain_hi:.1f}", YELLOW)

        rows.append(
            f"{name:14s}{v:9.2f}{model[i]:9.2f}"
            f"{lo[i]:9.2f}..{hi[i]:8.2f}{gl[i]:9.2f}..{gh[i]:8.2f}   {act}"
        )

    all_ok = ready == controllable
    rows.append("")
    if all_ok:
        rows.append(paint(
            f"READY: {ready}/{controllable} controllable joints are inside the aim range "
            f"(margin {margin:g} deg / {UNITS[5]} {0:g})", GREEN))
        rows.append("next: coordinate_bridge_20261113/predict_live_bridged.py  (read-only check)")
    else:
        rows.append(paint(
            f"{ready}/{controllable} ready.  turn: " + ", ".join(todo), YELLOW))
        rows.append(f"{DIM if color else ''}roll is passive and ignored; "
                    f"keep every joint inside 'aim for' to stay away from the model's limits"
                    f"{RESET if color else ''}")
    return "\n".join(rows), all_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--hz", type=float, default=8.0)
    ap.add_argument("--margin", type=float, default=10.0,
                    help="degrees of margin inside the window (arm joints)")
    ap.add_argument("--gripper-margin", type=float, default=5.0,
                    help="percentage points of margin for the gripper")
    ap.add_argument("--model-dir", default=str(HERE / "model"))
    ap.add_argument("--plain", action="store_true", help="no ANSI cursor control")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after N seconds (0 = run until Ctrl+C)")
    args = ap.parse_args()

    sys.path.insert(0, str(LEROBOT_SRC))
    sys.path.insert(0, str(POLICY_DIR))
    sys.path.insert(0, str(HERE / "coordinate_bridge_20261113"))
    from joint_frame_bridge import arm_to_model
    from readonly_bus import ReadOnlyArm, validate_calibration

    calibration = validate_calibration(CAL_PATH, REF_CAL_PATH)
    lo, hi, gl, gh, t_lo, t_hi = load_windows(
        Path(args.model_dir), calibration, args.margin, args.gripper_margin
    )

    arm = ReadOnlyArm(args.port, calibration)
    arm.connect()
    period = 1.0 / max(args.hz, 0.5)
    use_color = (not args.no_color) and (not args.plain or sys.stdout.isatty())
    last_ok = None
    started = time.perf_counter()
    try:
        if not args.plain:
            sys.stdout.write("\x1b[?25l")
        while True:
            if args.seconds and time.perf_counter() - started >= args.seconds:
                break
            t0 = time.perf_counter()
            local = np.asarray(arm.read(), dtype=np.float64)
            model = arm_to_model(local)
            text, ok = render(local, model, lo, hi, gl, gh, t_lo, t_hi, args.margin, use_color)
            if args.plain:
                if ok != last_ok:
                    print(text, flush=True)
                    print()
            else:
                sys.stdout.write("\x1b[H" + text + "\x1b[J")
                sys.stdout.flush()
            if ok and last_ok is not True:
                sys.stdout.write("\a")
            last_ok = ok
            sleep = period - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n\nstopped by user")
    finally:
        if not args.plain:
            sys.stdout.write("\x1b[?25h\n")
        arm.close()
        rep = arm.report()
        print(json.dumps({
            "blocked_packets": rep.get("blocked_packets"),
            "allowed_packet_counts": rep.get("allowed_packet_counts"),
            "torque_unchanged": arm.torque_before == arm.torque_after,
        }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
